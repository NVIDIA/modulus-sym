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
from sympy import Symbol, sin
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.dataset import DictImportanceSampledPointwiseIterableDataset
from modulus.sym.domain.constraint.utils import _compute_outvar
from modulus.sym.geometry.parameterization import Bounds


def test_DictImportanceSampledPointwiseIterableDataset():
    """sample sin function on a rectangle with importance measure sqrt(x**2 + y**2) and check its integral is zero"""
    paddle.seed(seed=123)
    np.random.seed(123)
    rec = Rectangle((-0.5, -0.5), (0.5, 0.5))
    invar = rec.sample_interior(
        100000, bounds=Bounds({Symbol("x"): (-0.5, 0.5), Symbol("y"): (-0.5, 0.5)})
    )
    outvar = _compute_outvar(invar, {"u": sin(2 * np.pi * Symbol("x") / 0.5)})

    def importance_measure(invar):
        return (invar["x"] ** 2 + invar["y"] ** 2) ** 0.5 + 0.01

    dataset = DictImportanceSampledPointwiseIterableDataset(
        invar=invar,
        outvar=outvar,
        batch_size=10000,
        importance_measure=importance_measure,
    )
    invar, outvar, lambda_weighting = next(iter(dataset))
    assert np.isclose(
        paddle.sum(x=outvar["u"] * invar["area"]), 0.0, rtol=0.01, atol=0.01
    )


if __name__ == "__main__":
    test_DictImportanceSampledPointwiseIterableDataset()
