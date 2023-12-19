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
import itertools
from modulus.sym.key import Key
from modulus.sym.models.super_res_net import SRResNetArch


def test_srresnet():
    model = SRResNetArch(
        input_keys=[Key("x", size=4)],
        output_keys=[Key("y", size=4), Key("z", size=2)],
        n_resid_blocks=8,
        scaling_factor=8,
    )
    bsize = 4
    x = {"x": paddle.randn(shape=(bsize, 4, 32, 20, 8))}
    outvar = model.forward(x)
    assert outvar["y"].shape == (bsize, 4, 256, 160, 64)
    assert outvar["z"].shape == (bsize, 2, 256, 160, 64)
    model = SRResNetArch(
        input_keys=[Key("x", size=4)],
        output_keys=[Key("y", size=3), Key("z", size=1)],
        n_resid_blocks=8,
        scaling_factor=2,
    )
    bsize = 2
    x = {"x": paddle.randn(shape=(bsize, 4, 24, 24, 20))}
    outvar = model.forward(x)
    assert outvar["y"].shape == (bsize, 3, 48, 48, 40)
    assert outvar["z"].shape == (bsize, 1, 48, 48, 40)
    model = SRResNetArch(
        input_keys=[Key("x", size=4)],
        output_keys=[Key("y", size=3), Key("z", size=3)],
        n_resid_blocks=8,
        scaling_factor=2,
    )
    bsize = 5
    x = {"x": paddle.randn(shape=(bsize, 4, 16, 16, 32))}
    outvar = model.forward(x)
    assert outvar["y"].shape == (bsize, 3, 32, 32, 64)
    assert outvar["z"].shape == (bsize, 3, 32, 32, 64)
