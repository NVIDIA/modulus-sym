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
import os
from modulus.sym.eq.pdes.signed_distance_function import ScreenedPoissonDistance


def test_screened_poisson_distance_equation():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    distance = np.exp(x + y + z)
    distance__x = np.exp(x + y + z)
    distance__y = np.exp(x + y + z)
    distance__z = np.exp(x + y + z)
    distance__x__x = np.exp(x + y + z)
    distance__y__y = np.exp(x + y + z)
    distance__z__z = np.exp(x + y + z)
    tau = 0.1
    sdf_grad = 1 - distance__x**2 - distance__y**2 - distance__z**2
    poisson = np.sqrt(tau) * (distance__x__x + distance__y__y + distance__z__z)
    screened_poisson_distance_true = sdf_grad + poisson
    screened_poisson_distance_eq = ScreenedPoissonDistance(
        distance="distance", tau=tau, dim=3
    )
    evaluations = screened_poisson_distance_eq.make_nodes()[0].evaluate(
        {
            "distance__x": paddle.to_tensor(data=distance__x, dtype="float32"),
            "distance__y": paddle.to_tensor(data=distance__y, dtype="float32"),
            "distance__z": paddle.to_tensor(data=distance__z, dtype="float32"),
            "distance__x__x": paddle.to_tensor(data=distance__x__x, dtype="float32"),
            "distance__y__y": paddle.to_tensor(data=distance__y__y, dtype="float32"),
            "distance__z__z": paddle.to_tensor(data=distance__z__z, dtype="float32"),
        }
    )
    screened_poisson_distance_eq_eval_pred = evaluations[
        "screened_poisson_distance"
    ].numpy()
    assert np.allclose(
        screened_poisson_distance_eq_eval_pred, screened_poisson_distance_true
    ), "Test Failed!"


if __name__ == "__main__":
    test_screened_poisson_distance_equation()
