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
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
import numpy as np
import os


def test_zero_equation():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    u = np.exp(2 * x + y + z + t)
    v = np.exp(x + 2 * y + z + t)
    w = np.exp(x + y + 2 * z + t)
    u__x = 2 * np.exp(2 * x + y + z + t)
    u__y = 1 * np.exp(2 * x + y + z + t)
    u__z = 1 * np.exp(2 * x + y + z + t)
    v__x = 1 * np.exp(x + 2 * y + z + t)
    v__y = 2 * np.exp(x + 2 * y + z + t)
    v__z = 1 * np.exp(x + 2 * y + z + t)
    w__x = 1 * np.exp(x + y + 2 * z + t)
    w__y = 1 * np.exp(x + y + 2 * z + t)
    w__z = 2 * np.exp(x + y + 2 * z + t)
    normal_distance = np.exp(x + y + z)
    rho = 1.0
    nu = 0.2
    max_distance = 0.5
    mixing_length = np.minimum(0.419 * normal_distance, 0.09 * max_distance)
    G = (
        2 * u__x**2
        + 2 * v__y**2
        + 2 * w__z**2
        + (u__y + v__x) ** 2
        + (u__z + w__x) ** 2
        + (v__z + w__y) ** 2
    )
    nu_true = nu + rho * mixing_length**2 * np.sqrt(G)
    zero_eq = ZeroEquation(nu=nu, max_distance=max_distance, rho=rho, dim=3, time=True)
    evaluations_zero_eq = zero_eq.make_nodes()[0].evaluate(
        {
            "u__x": paddle.to_tensor(data=u__x, dtype="float32"),
            "u__y": paddle.to_tensor(data=u__y, dtype="float32"),
            "u__z": paddle.to_tensor(data=u__z, dtype="float32"),
            "v__x": paddle.to_tensor(data=v__x, dtype="float32"),
            "v__y": paddle.to_tensor(data=v__y, dtype="float32"),
            "v__z": paddle.to_tensor(data=v__z, dtype="float32"),
            "w__x": paddle.to_tensor(data=w__x, dtype="float32"),
            "w__y": paddle.to_tensor(data=w__y, dtype="float32"),
            "w__z": paddle.to_tensor(data=w__z, dtype="float32"),
            "sdf": paddle.to_tensor(data=normal_distance, dtype="float32"),
        }
    )
    zero_eq_eval_pred = evaluations_zero_eq["nu"].numpy()
    assert np.allclose(zero_eq_eval_pred, nu_true), "Test Failed!"


if __name__ == "__main__":
    test_zero_equation()
