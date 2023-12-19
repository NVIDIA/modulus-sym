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

from sympy import Symbol
import numpy as np
from pathlib import Path
from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.geometry import Parameterization

dir_path = Path(__file__).parent


def test_tesselated_geometry():
    cube = Tessellation.from_stl(dir_path / "stls/cube.stl")
    boundary = cube.sample_boundary(
        1000, parameterization=Parameterization({Symbol("fake_param"): 1})
    )
    interior = cube.sample_interior(
        1000, parameterization=Parameterization({Symbol("fake_param"): 1})
    )
    assert np.isclose(np.sum(boundary["area"]), 6.0)
    assert np.isclose(np.sum(interior["area"]), 1.0)
