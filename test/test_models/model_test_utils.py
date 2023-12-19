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
from modulus.sym.graph import Graph
from modulus.sym.key import Key
from modulus.sym.models.arch import FuncArch, Arch
from typing import List

_ = paddle.seed(seed=0)
device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
    "cuda", "gpu"
)


def validate_func_arch_net(
    ref_net: Arch, deriv_keys: List[Key], validate_with_dict_forward: bool
):
    """
    Using double precision for testing.
    """
    if validate_with_dict_forward:
        ref_net.forward = ref_net._dict_forward
    ref_graph = (
        Graph(
            [ref_net.make_node("ref_net", jit=False)],
            ref_net.input_keys,
            deriv_keys + ref_net.output_keys,
            func_arch=False,
        )
        .double()
        .to(device)
    )
    ft_net = FuncArch(arch=ref_net, deriv_keys=deriv_keys).double().to(device)
    batch_size = 20
    out_52 = paddle.rand(shape=[batch_size, v.size], dtype="float64")
    out_52.stop_gradient = not True
    in_vars = {v.name: out_52 for v in ref_net.input_keys}
    ft_out = ft_net(in_vars)
    ref_out = ref_graph(in_vars)
    for k in ref_out.keys():
        assert paddle.allclose(x=ref_out[k], y=ft_out[k]).item()
    return ft_net
