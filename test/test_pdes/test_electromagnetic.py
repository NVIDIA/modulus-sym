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
from modulus.sym.eq.pdes.electromagnetic import MaxwellFreqReal, SommerfeldBC, PEC
import numpy as np
import os


def test_maxwell_freq_real():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    ux = np.exp(1 * x + 1 * y + 1 * z)
    uy = np.exp(2 * x + 2 * y + 2 * z)
    uz = np.exp(3 * x + 3 * y + 3 * z)
    ux__x = 1 * np.exp(1 * x + 1 * y + 1 * z)
    uy__x = 2 * np.exp(2 * x + 2 * y + 2 * z)
    uz__x = 3 * np.exp(3 * x + 3 * y + 3 * z)
    ux__y = 1 * np.exp(1 * x + 1 * y + 1 * z)
    uy__y = 2 * np.exp(2 * x + 2 * y + 2 * z)
    uz__y = 3 * np.exp(3 * x + 3 * y + 3 * z)
    ux__z = 1 * np.exp(1 * x + 1 * y + 1 * z)
    uy__z = 2 * np.exp(2 * x + 2 * y + 2 * z)
    uz__z = 3 * np.exp(3 * x + 3 * y + 3 * z)
    ux__x__x = 1 * np.exp(1 * x + 1 * y + 1 * z)
    ux__x__y = 1 * np.exp(1 * x + 1 * y + 1 * z)
    ux__x__z = 1 * np.exp(1 * x + 1 * y + 1 * z)
    ux__y__x = ux__x__y
    ux__y__y = 1 * np.exp(1 * x + 1 * y + 1 * z)
    ux__y__z = 1 * np.exp(1 * x + 1 * y + 1 * z)
    ux__z__x = ux__x__z
    ux__z__y = ux__y__z
    ux__z__z = 1 * np.exp(1 * x + 1 * y + 1 * z)
    uy__x__x = 4 * np.exp(2 * x + 2 * y + 2 * z)
    uy__x__y = 4 * np.exp(2 * x + 2 * y + 2 * z)
    uy__x__z = 4 * np.exp(2 * x + 2 * y + 2 * z)
    uy__y__x = uy__x__y
    uy__y__y = 4 * np.exp(2 * x + 2 * y + 2 * z)
    uy__y__z = 4 * np.exp(2 * x + 2 * y + 2 * z)
    uy__z__x = uy__x__z
    uy__z__y = uy__y__z
    uy__z__z = 4 * np.exp(2 * x + 2 * y + 2 * z)
    uz__x__x = 9 * np.exp(3 * x + 3 * y + 3 * z)
    uz__x__y = 9 * np.exp(3 * x + 3 * y + 3 * z)
    uz__x__z = 9 * np.exp(3 * x + 3 * y + 3 * z)
    uz__y__x = uz__x__y
    uz__y__y = 9 * np.exp(3 * x + 3 * y + 3 * z)
    uz__y__z = 9 * np.exp(3 * x + 3 * y + 3 * z)
    uz__z__x = uz__x__z
    uz__z__y = uz__y__z
    uz__z__z = 9 * np.exp(3 * x + 3 * y + 3 * z)
    curlux = uz__y - uy__z
    curluy = ux__z - uz__x
    curluz = uy__x - ux__y
    curlcurlux = (
        4 * np.exp(2 * x + 2 * y + 2 * z)
        - 1 * np.exp(1 * x + 1 * y + 1 * z)
        - 1 * np.exp(1 * x + 1 * y + 1 * z)
        + 9 * np.exp(3 * x + 3 * y + 3 * z)
    )
    curlcurluy = (
        9 * np.exp(3 * x + 3 * y + 3 * z)
        - 4 * np.exp(2 * x + 2 * y + 2 * z)
        - 4 * np.exp(2 * x + 2 * y + 2 * z)
        + 1 * np.exp(1 * x + 1 * y + 1 * z)
    )
    curlcurluz = (
        1 * np.exp(1 * x + 1 * y + 1 * z)
        - 9 * np.exp(3 * x + 3 * y + 3 * z)
        - 9 * np.exp(3 * x + 3 * y + 3 * z)
        + 4 * np.exp(2 * x + 2 * y + 2 * z)
    )
    k = 0.1
    Maxwell_Freq_real_x_true = curlcurlux - k**2 * ux
    Maxwell_Freq_real_y_true = curlcurluy - k**2 * uy
    Maxwell_Freq_real_z_true = curlcurluz - k**2 * uz
    maxwell_eq = MaxwellFreqReal(k=k)
    evaluations_MaxwellFreqReal_x = maxwell_eq.make_nodes()[0].evaluate(
        {
            "ux": paddle.to_tensor(data=ux, dtype="float32"),
            "uy__x__y": paddle.to_tensor(data=uy__x__y, dtype="float32"),
            "ux__y__y": paddle.to_tensor(data=ux__y__y, dtype="float32"),
            "ux__z__z": paddle.to_tensor(data=ux__z__z, dtype="float32"),
            "uz__x__z": paddle.to_tensor(data=uz__x__z, dtype="float32"),
        }
    )
    evaluations_MaxwellFreqReal_y = maxwell_eq.make_nodes()[1].evaluate(
        {
            "uy": paddle.to_tensor(data=uy, dtype="float32"),
            "uz__y__z": paddle.to_tensor(data=uz__y__z, dtype="float32"),
            "uy__z__z": paddle.to_tensor(data=uy__z__z, dtype="float32"),
            "uy__x__x": paddle.to_tensor(data=uy__x__x, dtype="float32"),
            "ux__x__y": paddle.to_tensor(data=ux__x__y, dtype="float32"),
        }
    )
    evaluations_MaxwellFreqReal_z = maxwell_eq.make_nodes()[2].evaluate(
        {
            "uz": paddle.to_tensor(data=uz, dtype="float32"),
            "ux__x__z": paddle.to_tensor(data=ux__x__z, dtype="float32"),
            "uz__x__x": paddle.to_tensor(data=uz__x__x, dtype="float32"),
            "uz__y__y": paddle.to_tensor(data=uz__y__y, dtype="float32"),
            "uy__y__z": paddle.to_tensor(data=uy__y__z, dtype="float32"),
        }
    )
    Maxwell_Freq_real_x_eval_pred = evaluations_MaxwellFreqReal_x[
        "Maxwell_Freq_real_x"
    ].numpy()
    Maxwell_Freq_real_y_eval_pred = evaluations_MaxwellFreqReal_y[
        "Maxwell_Freq_real_y"
    ].numpy()
    Maxwell_Freq_real_z_eval_pred = evaluations_MaxwellFreqReal_z[
        "Maxwell_Freq_real_z"
    ].numpy()
    assert np.allclose(
        Maxwell_Freq_real_x_eval_pred, Maxwell_Freq_real_x_true
    ), "Test Failed!"
    assert np.allclose(
        Maxwell_Freq_real_y_eval_pred, Maxwell_Freq_real_y_true
    ), "Test Failed!"
    assert np.allclose(
        Maxwell_Freq_real_z_eval_pred, Maxwell_Freq_real_z_true
    ), "Test Failed!"


def test_sommerfeld_bc():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    normal_x = np.random.rand(1024, 1)
    normal_y = np.random.rand(1024, 1)
    normal_z = np.random.rand(1024, 1)
    ux = np.exp(1 * x + 1 * y + 1 * z)
    uy = np.exp(2 * x + 2 * y + 2 * z)
    uz = np.exp(3 * x + 3 * y + 3 * z)
    ux__x = 1 * np.exp(1 * x + 1 * y + 1 * z)
    uy__x = 2 * np.exp(2 * x + 2 * y + 2 * z)
    uz__x = 3 * np.exp(3 * x + 3 * y + 3 * z)
    ux__y = 1 * np.exp(1 * x + 1 * y + 1 * z)
    uy__y = 2 * np.exp(2 * x + 2 * y + 2 * z)
    uz__y = 3 * np.exp(3 * x + 3 * y + 3 * z)
    ux__z = 1 * np.exp(1 * x + 1 * y + 1 * z)
    uy__z = 2 * np.exp(2 * x + 2 * y + 2 * z)
    uz__z = 3 * np.exp(3 * x + 3 * y + 3 * z)
    curlux = uz__y - uy__z
    curluy = ux__z - uz__x
    curluz = uy__x - ux__y
    SommerfeldBC_real_x_true = normal_y * curluz - normal_z * curluy
    SommerfeldBC_real_y_true = normal_z * curlux - normal_x * curluz
    SommerfeldBC_real_z_true = normal_x * curluy - normal_y * curlux
    sommerfeld_bc = SommerfeldBC()
    evaluations_SommerfeldBC_real_x = sommerfeld_bc.make_nodes()[0].evaluate(
        {
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "normal_z": paddle.to_tensor(data=normal_z, dtype="float32"),
            "ux__y": paddle.to_tensor(data=ux__y, dtype="float32"),
            "uy__x": paddle.to_tensor(data=uy__x, dtype="float32"),
            "ux__z": paddle.to_tensor(data=ux__z, dtype="float32"),
            "uz__x": paddle.to_tensor(data=uz__x, dtype="float32"),
        }
    )
    evaluations_SommerfeldBC_real_y = sommerfeld_bc.make_nodes()[1].evaluate(
        {
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_z": paddle.to_tensor(data=normal_z, dtype="float32"),
            "ux__y": paddle.to_tensor(data=ux__y, dtype="float32"),
            "uy__x": paddle.to_tensor(data=uy__x, dtype="float32"),
            "uy__z": paddle.to_tensor(data=uy__z, dtype="float32"),
            "uz__y": paddle.to_tensor(data=uz__y, dtype="float32"),
        }
    )
    evaluations_SommerfeldBC_real_z = sommerfeld_bc.make_nodes()[2].evaluate(
        {
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "ux__z": paddle.to_tensor(data=ux__z, dtype="float32"),
            "uz__x": paddle.to_tensor(data=uz__x, dtype="float32"),
            "uy__z": paddle.to_tensor(data=uy__z, dtype="float32"),
            "uz__y": paddle.to_tensor(data=uz__y, dtype="float32"),
        }
    )
    SommerfeldBC_real_x_eval_pred = evaluations_SommerfeldBC_real_x[
        "SommerfeldBC_real_x"
    ].numpy()
    SommerfeldBC_real_y_eval_pred = evaluations_SommerfeldBC_real_y[
        "SommerfeldBC_real_y"
    ].numpy()
    SommerfeldBC_real_z_eval_pred = evaluations_SommerfeldBC_real_z[
        "SommerfeldBC_real_z"
    ].numpy()
    assert np.allclose(
        SommerfeldBC_real_x_eval_pred, SommerfeldBC_real_x_true, atol=0.0001
    ), "Test Failed!"
    assert np.allclose(
        SommerfeldBC_real_y_eval_pred, SommerfeldBC_real_y_true, atol=0.0001
    ), "Test Failed!"
    assert np.allclose(
        SommerfeldBC_real_z_eval_pred, SommerfeldBC_real_z_true, atol=0.0001
    ), "Test Failed!"


def test_pec():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    normal_x = np.random.rand(1024, 1)
    normal_y = np.random.rand(1024, 1)
    normal_z = np.random.rand(1024, 1)
    ux = np.exp(1 * x + 1 * y + 1 * z)
    uy = np.exp(2 * x + 2 * y + 2 * z)
    uz = np.exp(3 * x + 3 * y + 3 * z)
    PEC_x_true = normal_y * uz - normal_z * uy
    PEC_y_true = normal_z * ux - normal_x * uz
    PEC_z_true = normal_x * uy - normal_y * ux
    pec = PEC()
    evaluations_PEC_x = pec.make_nodes()[0].evaluate(
        {
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "normal_z": paddle.to_tensor(data=normal_z, dtype="float32"),
            "uz": paddle.to_tensor(data=uz, dtype="float32"),
            "uy": paddle.to_tensor(data=uy, dtype="float32"),
        }
    )
    evaluations_PEC_y = pec.make_nodes()[1].evaluate(
        {
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_z": paddle.to_tensor(data=normal_z, dtype="float32"),
            "ux": paddle.to_tensor(data=ux, dtype="float32"),
            "uz": paddle.to_tensor(data=uz, dtype="float32"),
        }
    )
    evaluations_PEC_z = pec.make_nodes()[2].evaluate(
        {
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "ux": paddle.to_tensor(data=ux, dtype="float32"),
            "uy": paddle.to_tensor(data=uy, dtype="float32"),
        }
    )
    PEC_x_eval_pred = evaluations_PEC_x["PEC_x"].numpy()
    PEC_y_eval_pred = evaluations_PEC_y["PEC_y"].numpy()
    PEC_z_eval_pred = evaluations_PEC_z["PEC_z"].numpy()
    assert np.allclose(PEC_x_eval_pred, PEC_x_true, atol=0.0001), "Test Failed!"
    assert np.allclose(PEC_y_eval_pred, PEC_y_true, atol=0.0001), "Test Failed!"
    assert np.allclose(PEC_z_eval_pred, PEC_z_true, atol=0.0001), "Test Failed!"


if __name__ == "__main__":
    test_maxwell_freq_real()
    test_sommerfeld_bc()
    test_pec()
