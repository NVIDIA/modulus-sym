# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from modulus.sym.models.deeponet import DeepONetArch
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.models.pix2pix import Pix2PixArch
import numpy as np
from modulus.sym.key import Key
import pytest
from modulus.sym.graph import Graph
from modulus.sym.models.arch import FuncArch
from .model_test_utils import validate_func_arch_net

_ = paddle.seed(seed=0)
device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
    "cuda", "gpu"
)
# >>>torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize(
    "branch_input_keys", [[Key("a", 100)], [Key("a", 100, scale=(1.0, 2.0))]]
)
@pytest.mark.parametrize("validate_with_dict_forward", [True, False])
@pytest.mark.parametrize("dim", [1, 2])
def test_func_arch_deeponet(branch_input_keys, validate_with_dict_forward, dim):
    deriv_keys = [Key.from_str("u__x"), Key.from_str("u__x__x")]
    branch_net = FullyConnectedArch(
        input_keys=branch_input_keys,
        output_keys=[Key("branch", 128)],
        nr_layers=4,
        layer_size=128,
    )
    trunk_net = FourierNetArch(
        input_keys=[Key("x", 1)],
        output_keys=[Key("trunk", 128)],
        nr_layers=4,
        layer_size=128,
        frequencies=("axis", [i for i in range(5)]),
    )
    ref_net = DeepONetArch(
        branch_net=branch_net, trunk_net=trunk_net, output_keys=[Key("u")]
    )
    validate_func_arch_net(ref_net, deriv_keys, validate_with_dict_forward)


@pytest.mark.parametrize("validate_with_dict_forward", [True, False])
def test_func_arch_deeponet_with_pix2pix(validate_with_dict_forward):
    """
    deeponet does not support FuncArch if branch_net is Pix2PixArch.
    """
    deriv_keys = [Key.from_str("sol__x"), Key.from_str("sol__x__x")]
    branch_input_keys = [Key("coeff")]
    output_keys = [Key("sol")]
    branch_net = Pix2PixArch(
        input_keys=branch_input_keys,
        output_keys=[Key("branch")],
        dimension=2,
        conv_layer_size=32,
    )
    trunk_net = FourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("trunk", 256)],
        nr_layers=5,
        layer_size=128,
        frequencies=("axis", [i for i in range(5)]),
    )
    ref_net = DeepONetArch(
        branch_net=branch_net,
        trunk_net=trunk_net,
        output_keys=output_keys,
        branch_dim=1024,
    )
    if validate_with_dict_forward:
        ref_net.forward = ref_net._dict_forward
    ref_graph = Graph(
        [ref_net.make_node("ref_net", jit=False)],
        ref_net.input_keys,
        deriv_keys + [Key("sol")],
        func_arch=False,
    ).to(device)
    assert not ref_net.supports_func_arch
    ft_graph = Graph(
        [ref_net.make_node("ref_net", jit=False)],
        ref_net.input_keys,
        deriv_keys + [Key("sol")],
        func_arch=True,
    ).to(device)
    for node in ft_graph.node_evaluation_order:
        evaluate = node.evaluate
        assert not isinstance(evaluate, FuncArch)
    out_32 = paddle.rand(shape=[100, 1])
    out_32.stop_gradient = not True
    x = out_32
    out_33 = paddle.rand(shape=[100, 1])
    out_33.stop_gradient = not True
    y = out_33
    out_34 = paddle.rand(shape=[100, branch_input_keys[0].size, 32, 32])
    out_34.stop_gradient = not True
    coeff = out_34
    in_vars = {"x": x, "y": y, "coeff": coeff}
    ft_out = ft_graph(in_vars)
    ref_out = ref_graph(in_vars)
    for k in ref_out.keys():
        assert paddle.allclose(x=ref_out[k], y=ft_out[k], atol=6e-05).item()
