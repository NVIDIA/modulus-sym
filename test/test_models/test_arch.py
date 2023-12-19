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
import numpy as np
from modulus.sym.constants import diff
from modulus.sym.key import Key
from modulus.sym.models.arch import Arch

paddle.seed(seed=0)
device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
    "cuda", "gpu"
)


def test_slice_input():
    x = paddle.rand(shape=[100, 1])
    y = paddle.rand(shape=[100, 2])
    z = paddle.rand(shape=[100, 1])
    input_variables = {"x": x, "y": y, "z": z}
    input_keys = [Key("x", 1), Key("y", 2), Key("z", 1)]
    input_key_dict = {str(var): var.size for var in input_keys}
    ipt = Arch.prepare_input(input_variables, input_key_dict.keys(), {}, dim=-1)
    slice_keys = ["x", "z"]
    expected = Arch.prepare_input(input_variables, slice_keys, {}, dim=-1)
    slice_index = Arch.prepare_slice_index(input_key_dict, slice_keys)
    result = Arch.slice_input(ipt, slice_index, dim=-1)
    assert paddle.allclose(x=result, y=expected).item()
    slice_keys = ["y", "z"]
    expected = Arch.prepare_input(input_variables, slice_keys, {}, dim=-1)
    slice_index = Arch.prepare_slice_index(input_key_dict, slice_keys)
    result = Arch.slice_input(ipt, slice_index, dim=-1)
    assert paddle.allclose(x=result, y=expected).item()


def validate_process_input_output(input_variables, arch):
    expected = Arch.prepare_input(
        input_variables,
        arch.input_key_dict.keys(),
        {},
        dim=-1,
        input_scales=arch.input_scales,
        periodicity=arch.periodicity,
    )
    result = Arch.concat_input(input_variables, arch.input_key_dict.keys(), {}, dim=-1)
    result = Arch.process_input(
        result, arch.input_scales_tensor, arch.periodicity, arch.input_key_dict, dim=-1
    )
    assert paddle.allclose(x=expected, y=result).item()
    batch_size, output_size = expected.shape[0], sum(arch.output_key_dict.values())
    y = paddle.rand(shape=[batch_size, output_size])
    expected = Arch.prepare_output(
        y, arch.output_key_dict, dim=-1, output_scales=arch.output_scales
    )
    result = Arch.process_output(y, output_scales_tensor=arch.output_scales_tensor)
    result = Arch.split_output(result, output_dict=arch.output_key_dict, dim=-1)
    assert expected.keys() == result.keys()
    for key in expected:
        assert paddle.allclose(x=expected[key], y=result[key]).item()


def test_process_input_output():
    x = paddle.ones(shape=[100, 1])
    y = paddle.ones(shape=[100, 2])
    z = paddle.ones(shape=[100, 1])
    input_variables = {"x": x, "y": y, "z": z}
    input_keys = [Key("x", 1), Key("y", 2), Key("z", 1)]
    output_keys = [Key("u", 1), Key("v", 1)]
    arch = Arch(input_keys, output_keys)
    validate_process_input_output(input_variables, arch)
    assert arch.input_scales_tensor is None
    assert arch.output_scales_tensor is None
    input_keys = [
        Key("x", 1, scale=(0.0, 1.0)),
        Key("y", 2, scale=(0.0, 2.0)),
        Key("z", 1, scale=(0.0, 3.0)),
    ]
    output_keys = [Key("u", 1, scale=(1.0, 2.0)), Key("v", 1)]
    arch = Arch(input_keys, output_keys)
    validate_process_input_output(input_variables, arch)
    assert paddle.allclose(
        x=arch.input_scales_tensor,
        y=paddle.to_tensor(data=[[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 2.0, 3.0]]),
    ).item()
    assert paddle.allclose(
        x=arch.output_scales_tensor, y=paddle.to_tensor(data=[[1.0, 0.0], [2.0, 1.0]])
    ).item()
    arch = Arch(
        input_keys,
        output_keys,
        periodicity={"x": (0.0, 2 * np.pi), "y": (np.pi, 4 * np.pi)},
    )
    validate_process_input_output(input_variables, arch)


if __name__ == "__main__":
    test_slice_input()
    test_process_input_output()
