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
from modulus.sym.models.fully_connected import FullyConnectedArch
import numpy as np
from pathlib import Path
from modulus.sym.key import Key
import pytest
from .model_test_utils import validate_func_arch_net

dir_path = Path(__file__).parent


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases"), ("weight_g", "alphas")]
    for i in range(nr_layers):
        for pt_name, tf_name in names:
            _dict["_impl.layers." + str(i) + ".linear." + pt_name] = (
                "fc" + str(i) + "/" + tf_name + ":0"
            )
    for pt_name, tf_name in names[:2]:
        _dict["_impl.final_layer.linear." + pt_name] = "fc_final/" + tf_name + ":0"
    return _dict


@pytest.mark.parametrize("jit", [False])
def test_fully_connected(jit):
    filename = dir_path / "data/test_fully_connected.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    arch = FullyConnectedArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        layer_size=params["layer_size"],
        nr_layers=params["nr_layers"],
    )
    name_dict = make_dict(params["nr_layers"])
    for _name, _tensor in arch.named_parameters():
        if not _tensor.stop_gradient:
            _tensor.data = paddle.to_tensor(data=Wbs[name_dict[_name]].T)
    data_out2 = arch(
        {
            "x": paddle.to_tensor(data=data_in[:, 0:1]),
            "y": paddle.to_tensor(data=data_in[:, 1:2]),
        }
    )
    data_out2 = data_out2["u"].detach().numpy()
    data_out1 = test_data["data_out"]
    assert np.allclose(data_out1, data_out2, rtol=0.001), "Test failed!"
    print("Success!")


def validate_func_arch_fully_connected(
    input_keys, output_keys, periodicity, deriv_keys, validate_with_dict_forward
):
    ref_net = FullyConnectedArch(
        input_keys=input_keys,
        output_keys=output_keys,
        periodicity=periodicity,
        nr_layers=3,
    )
    ft_net = validate_func_arch_net(ref_net, deriv_keys, validate_with_dict_forward)
    return ft_net


@pytest.mark.parametrize(
    "input_keys",
    [
        [Key("x"), Key("y")],
        [Key("x"), Key("y", scale=(1.0, 2.0))],
        [Key("x"), Key("z", size=100), Key("y")],
    ],
)
@pytest.mark.parametrize(
    "output_keys",
    [
        [Key("u"), Key("v"), Key("p")],
        [Key("u"), Key("v"), Key("p", scale=(1.0, 2.0)), Key("w", size=100)],
    ],
)
@pytest.mark.parametrize(
    "periodicity",
    [{}, {"x": (0.0, 2 * np.pi)}, {"x": (0.0, 2 * np.pi), "y": (np.pi, 4 * np.pi)}],
)
@pytest.mark.parametrize("validate_with_dict_forward", [True, False])
def test_func_arch_fully_connected(
    input_keys, output_keys, periodicity, validate_with_dict_forward
):
    deriv_keys = [Key.from_str("u__x"), Key.from_str("v__y"), Key.from_str("p__x")]
    ft_net = validate_func_arch_fully_connected(
        input_keys, output_keys, periodicity, deriv_keys, validate_with_dict_forward
    )
    assert paddle.allclose(
        x=ft_net.needed_output_dims, y=paddle.to_tensor(data=[0, 1, 2])
    ).item()
    deriv_keys = [Key.from_str("u__x"), Key.from_str("p__x")]
    ft_net = validate_func_arch_fully_connected(
        input_keys, output_keys, periodicity, deriv_keys, validate_with_dict_forward
    )
    assert paddle.allclose(
        x=ft_net.needed_output_dims, y=paddle.to_tensor(data=[0, 2])
    ).item()
    deriv_keys = [Key.from_str("v__y")]
    ft_net = validate_func_arch_fully_connected(
        input_keys, output_keys, periodicity, deriv_keys, validate_with_dict_forward
    )
    assert paddle.allclose(
        x=ft_net.needed_output_dims, y=paddle.to_tensor(data=[1])
    ).item()
    deriv_keys = [
        Key.from_str("u__x__x"),
        Key.from_str("v__y__y"),
        Key.from_str("p__x__x"),
    ]
    ft_net = validate_func_arch_fully_connected(
        input_keys, output_keys, periodicity, deriv_keys, validate_with_dict_forward
    )
    assert paddle.allclose(
        x=ft_net.needed_output_dims, y=paddle.to_tensor(data=[0, 1, 2])
    ).item()
    deriv_keys = [
        Key.from_str("u__x__x"),
        Key.from_str("v__y__y"),
        Key.from_str("p__x"),
    ]
    ft_net = validate_func_arch_fully_connected(
        input_keys, output_keys, periodicity, deriv_keys, validate_with_dict_forward
    )
    assert paddle.allclose(
        x=ft_net.needed_output_dims, y=paddle.to_tensor(data=[0, 1, 2])
    ).item()
    deriv_keys = [Key.from_str("u__x__x"), Key.from_str("p__x__x")]
    ft_net = validate_func_arch_fully_connected(
        input_keys, output_keys, periodicity, deriv_keys, validate_with_dict_forward
    )
    assert paddle.allclose(
        x=ft_net.needed_output_dims, y=paddle.to_tensor(data=[0, 2])
    ).item()


if __name__ == "__main__":
    test_fully_connected(True)
    test_fully_connected(False)
