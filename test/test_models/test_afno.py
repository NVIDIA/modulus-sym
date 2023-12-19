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
from modulus.sym.models.afno import AFNOArch


def test_afno():
    model = AFNOArch(
        input_keys=[Key("x", size=2)],
        output_keys=[Key("u", size=2), Key("p")],
        img_shape=(240, 240),
        patch_size=16,
        embed_dim=256,
        depth=4,
        num_blocks=8,
    )
    node = model.make_node(name="AFNO", jit=False)
    bsize = 5
    invar = {"x": paddle.randn(shape=[bsize, 2, 240, 240])}
    outvar = node.evaluate(invar)
    assert outvar["u"].shape == [bsize, 2, 240, 240]
    assert outvar["p"].shape == [bsize, 1, 240, 240]


test_afno()
