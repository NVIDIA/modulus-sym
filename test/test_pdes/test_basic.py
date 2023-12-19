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
from modulus.sym.eq.pdes.basic import GradNormal, Curl
import numpy as np
import os


def test_normal_gradient_equation():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    normal_x = np.random.rand(1024, 1)
    normal_y = np.random.rand(1024, 1)
    normal_z = np.random.rand(1024, 1)
    u = np.exp(2 * x + y + z + t)
    u__x = 2 * np.exp(2 * x + y + z + t)
    u__y = 1 * np.exp(2 * x + y + z + t)
    u__z = 1 * np.exp(2 * x + y + z + t)
    normal_gradient_u_true = normal_x * u__x + normal_y * u__y + normal_z * u__z
    normal_gradient_eq = GradNormal(T="u", dim=3, time=True)
    evaluations = normal_gradient_eq.make_nodes()[0].evaluate(
        {
            "u__x": paddle.to_tensor(data=u__x, dtype="float32"),
            "u__y": paddle.to_tensor(data=u__y, dtype="float32"),
            "u__z": paddle.to_tensor(data=u__z, dtype="float32"),
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "normal_z": paddle.to_tensor(data=normal_z, dtype="float32"),
        }
    )
    normal_gradient_u_eval_pred = evaluations["normal_gradient_u"].numpy()
    assert np.allclose(
        normal_gradient_u_eval_pred, normal_gradient_u_true
    ), "Test Failed!"


def test_curl():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    a = np.exp(2 * x + y + z)
    b = np.exp(x + 2 * y + z)
    c = np.exp(x + y + 2 * z)
    a__x = 2 * np.exp(2 * x + y + z)
    a__y = 1 * np.exp(2 * x + y + z)
    a__z = 1 * np.exp(2 * x + y + z)
    b__x = 1 * np.exp(x + 2 * y + z)
    b__y = 2 * np.exp(x + 2 * y + z)
    b__z = 1 * np.exp(x + 2 * y + z)
    c__x = 1 * np.exp(x + y + 2 * z)
    c__y = 1 * np.exp(x + y + 2 * z)
    c__z = 2 * np.exp(x + y + 2 * z)
    u_true = c__y - b__z
    v_true = a__z - c__x
    w_true = b__x - a__y
    curl_eq = Curl(("a", "b", "c"), ("u", "v", "w"))
    evaluations_u = curl_eq.make_nodes()[0].evaluate(
        {
            "c__y": paddle.to_tensor(data=c__y, dtype="float32"),
            "b__z": paddle.to_tensor(data=b__z, dtype="float32"),
        }
    )
    evaluations_v = curl_eq.make_nodes()[1].evaluate(
        {
            "a__z": paddle.to_tensor(data=a__z, dtype="float32"),
            "c__x": paddle.to_tensor(data=c__x, dtype="float32"),
        }
    )
    evaluations_w = curl_eq.make_nodes()[2].evaluate(
        {
            "b__x": paddle.to_tensor(data=b__x, dtype="float32"),
            "a__y": paddle.to_tensor(data=a__y, dtype="float32"),
        }
    )
    u_eval_pred = evaluations_u["u"].numpy()
    v_eval_pred = evaluations_v["v"].numpy()
    w_eval_pred = evaluations_w["w"].numpy()
    assert np.allclose(u_eval_pred, u_true, atol=0.0001), "Test Failed!"
    assert np.allclose(v_eval_pred, v_true, atol=0.0001), "Test Failed!"
    assert np.allclose(w_eval_pred, w_true, atol=0.0001), "Test Failed!"


if __name__ == "__main__":
    test_normal_gradient_equation()
    test_curl()
