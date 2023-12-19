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
from modulus.sym.eq.pdes.linear_elasticity import (
    LinearElasticity,
    LinearElasticityPlaneStress,
)
import numpy as np
import os


def test_linear_elasticity_equations():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    normal_x = np.random.rand(1024, 1)
    normal_y = np.random.rand(1024, 1)
    normal_z = np.random.rand(1024, 1)
    u = np.exp(2 * x + y + z + t)
    v = np.exp(x + 2 * y + z + t)
    w = np.exp(x + y + 2 * z + t)
    u__t__t = 1 * np.exp(2 * x + y + z + t)
    v__t__t = 1 * np.exp(x + 2 * y + z + t)
    w__t__t = 1 * np.exp(x + y + 2 * z + t)
    u__x = 2 * np.exp(2 * x + y + z + t)
    u__y = 1 * np.exp(2 * x + y + z + t)
    u__z = 1 * np.exp(2 * x + y + z + t)
    u__x__x = 2 * 2 * np.exp(2 * x + y + z + t)
    u__y__y = 1 * 1 * np.exp(2 * x + y + z + t)
    u__z__z = 1 * 1 * np.exp(2 * x + y + z + t)
    u__x__y = 1 * 2 * np.exp(2 * x + y + z + t)
    u__x__z = 1 * 2 * np.exp(2 * x + y + z + t)
    u__y__z = 1 * 1 * np.exp(2 * x + y + z + t)
    u__y__x = u__x__y
    u__z__x = u__x__z
    u__z__y = u__y__z
    v__x = 1 * np.exp(x + 2 * y + z + t)
    v__y = 2 * np.exp(x + 2 * y + z + t)
    v__z = 1 * np.exp(x + 2 * y + z + t)
    v__x__x = 1 * 1 * np.exp(x + 2 * y + z + t)
    v__y__y = 2 * 2 * np.exp(x + 2 * y + z + t)
    v__z__z = 1 * 1 * np.exp(x + 2 * y + z + t)
    v__x__y = 2 * 1 * np.exp(x + 2 * y + z + t)
    v__x__z = 1 * 1 * np.exp(x + 2 * y + z + t)
    v__y__z = 1 * 2 * np.exp(x + 2 * y + z + t)
    v__y__x = v__x__y
    v__z__x = v__x__z
    v__z__y = v__y__z
    w__x = 1 * np.exp(x + y + 2 * z + t)
    w__y = 1 * np.exp(x + y + 2 * z + t)
    w__z = 2 * np.exp(x + y + 2 * z + t)
    w__x__x = 1 * 1 * np.exp(x + y + 2 * z + t)
    w__y__y = 1 * 1 * np.exp(x + y + 2 * z + t)
    w__z__z = 2 * 2 * np.exp(x + y + 2 * z + t)
    w__x__y = 1 * 1 * np.exp(x + y + 2 * z + t)
    w__x__z = 2 * 1 * np.exp(x + y + 2 * z + t)
    w__y__z = 2 * 1 * np.exp(x + y + 2 * z + t)
    w__y__x = w__x__y
    w__z__x = w__x__z
    w__z__y = w__y__z
    sigma_xx = np.sin(x) * np.cos(y) * np.cos(z)
    sigma_yy = np.cos(x) * np.sin(y) * np.cos(z)
    sigma_zz = np.cos(x) * np.cos(y) * np.sin(z)
    sigma_xy = np.sin(x) * np.sin(y) * np.cos(z)
    sigma_xz = np.sin(x) * np.cos(y) * np.sin(z)
    sigma_yz = np.cos(x) * np.sin(y) * np.sin(z)
    sigma_xx__x = np.cos(x) * np.cos(y) * np.cos(z)
    sigma_yy__y = np.cos(x) * np.cos(y) * np.cos(z)
    sigma_zz__z = np.cos(x) * np.cos(y) * np.cos(z)
    sigma_xy__x = np.cos(x) * np.sin(y) * np.cos(z)
    sigma_xy__y = np.sin(x) * np.cos(y) * np.cos(z)
    sigma_xz__x = np.cos(x) * np.cos(y) * np.sin(z)
    sigma_xz__z = np.sin(x) * np.cos(y) * np.cos(z)
    sigma_yz__y = np.cos(x) * np.cos(y) * np.sin(z)
    sigma_yz__z = np.cos(x) * np.sin(y) * np.cos(z)
    E = 1.0
    nu = 0.1
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    rho = 10.0
    stress_disp_xx_true = lambda_ * (u__x + v__y + w__z) + 2 * mu * u__x - sigma_xx
    stress_disp_yy_true = lambda_ * (u__x + v__y + w__z) + 2 * mu * v__y - sigma_yy
    stress_disp_zz_true = lambda_ * (u__x + v__y + w__z) + 2 * mu * w__z - sigma_zz
    stress_disp_xy_true = mu * (u__y + v__x) - sigma_xy
    stress_disp_xz_true = mu * (u__z + w__x) - sigma_xz
    stress_disp_yz_true = mu * (v__z + w__y) - sigma_yz
    equilibrium_x_true = rho * u__t__t - (sigma_xx__x + sigma_xy__y + sigma_xz__z)
    equilibrium_y_true = rho * v__t__t - (sigma_xy__x + sigma_yy__y + sigma_yz__z)
    equilibrium_z_true = rho * w__t__t - (sigma_xz__x + sigma_yz__y + sigma_zz__z)
    traction_x_true = normal_x * sigma_xx + normal_y * sigma_xy + normal_z * sigma_xz
    traction_y_true = normal_x * sigma_xy + normal_y * sigma_yy + normal_z * sigma_yz
    traction_z_true = normal_x * sigma_xz + normal_y * sigma_yz + normal_z * sigma_zz
    navier_x_true = (
        rho * u__t__t
        - (lambda_ + mu) * (u__x__x + v__y__x + w__z__x)
        - mu * (u__x__x + u__y__y + u__z__z)
    )
    navier_y_true = (
        rho * v__t__t
        - (lambda_ + mu) * (u__x__y + v__y__y + w__z__y)
        - mu * (v__x__x + v__y__y + v__z__z)
    )
    navier_z_true = (
        rho * w__t__t
        - (lambda_ + mu) * (u__x__z + v__y__z + w__z__z)
        - mu * (w__x__x + w__y__y + w__z__z)
    )
    linear_elasticity_eq = LinearElasticity(nu=nu, E=E, rho=rho, dim=3, time=True)
    evaluations_stress_disp_xx = linear_elasticity_eq.make_nodes()[0].evaluate(
        {
            "u__x": paddle.to_tensor(data=u__x, dtype="float32"),
            "v__y": paddle.to_tensor(data=v__y, dtype="float32"),
            "w__z": paddle.to_tensor(data=w__z, dtype="float32"),
            "sigma_xx": paddle.to_tensor(data=sigma_xx, dtype="float32"),
        }
    )
    evaluations_stress_disp_yy = linear_elasticity_eq.make_nodes()[1].evaluate(
        {
            "u__x": paddle.to_tensor(data=u__x, dtype="float32"),
            "v__y": paddle.to_tensor(data=v__y, dtype="float32"),
            "w__z": paddle.to_tensor(data=w__z, dtype="float32"),
            "sigma_yy": paddle.to_tensor(data=sigma_yy, dtype="float32"),
        }
    )
    evaluations_stress_disp_zz = linear_elasticity_eq.make_nodes()[2].evaluate(
        {
            "u__x": paddle.to_tensor(data=u__x, dtype="float32"),
            "v__y": paddle.to_tensor(data=v__y, dtype="float32"),
            "w__z": paddle.to_tensor(data=w__z, dtype="float32"),
            "sigma_zz": paddle.to_tensor(data=sigma_zz, dtype="float32"),
        }
    )
    evaluations_stress_disp_xy = linear_elasticity_eq.make_nodes()[3].evaluate(
        {
            "u__y": paddle.to_tensor(data=u__y, dtype="float32"),
            "v__x": paddle.to_tensor(data=v__x, dtype="float32"),
            "sigma_xy": paddle.to_tensor(data=sigma_xy, dtype="float32"),
        }
    )
    evaluations_stress_disp_xz = linear_elasticity_eq.make_nodes()[4].evaluate(
        {
            "u__z": paddle.to_tensor(data=u__z, dtype="float32"),
            "w__x": paddle.to_tensor(data=w__x, dtype="float32"),
            "sigma_xz": paddle.to_tensor(data=sigma_xz, dtype="float32"),
        }
    )
    evaluations_stress_disp_yz = linear_elasticity_eq.make_nodes()[5].evaluate(
        {
            "v__z": paddle.to_tensor(data=v__z, dtype="float32"),
            "w__y": paddle.to_tensor(data=w__y, dtype="float32"),
            "sigma_yz": paddle.to_tensor(data=sigma_yz, dtype="float32"),
        }
    )
    evaluations_equilibrium_x = linear_elasticity_eq.make_nodes()[6].evaluate(
        {
            "u__t__t": paddle.to_tensor(data=u__t__t, dtype="float32"),
            "sigma_xx__x": paddle.to_tensor(data=sigma_xx__x, dtype="float32"),
            "sigma_xy__y": paddle.to_tensor(data=sigma_xy__y, dtype="float32"),
            "sigma_xz__z": paddle.to_tensor(data=sigma_xz__z, dtype="float32"),
        }
    )
    evaluations_equilibrium_y = linear_elasticity_eq.make_nodes()[7].evaluate(
        {
            "v__t__t": paddle.to_tensor(data=v__t__t, dtype="float32"),
            "sigma_xy__x": paddle.to_tensor(data=sigma_xy__x, dtype="float32"),
            "sigma_yy__y": paddle.to_tensor(data=sigma_yy__y, dtype="float32"),
            "sigma_yz__z": paddle.to_tensor(data=sigma_yz__z, dtype="float32"),
        }
    )
    evaluations_equilibrium_z = linear_elasticity_eq.make_nodes()[8].evaluate(
        {
            "w__t__t": paddle.to_tensor(data=w__t__t, dtype="float32"),
            "sigma_xz__x": paddle.to_tensor(data=sigma_xz__x, dtype="float32"),
            "sigma_yz__y": paddle.to_tensor(data=sigma_yz__y, dtype="float32"),
            "sigma_zz__z": paddle.to_tensor(data=sigma_zz__z, dtype="float32"),
        }
    )
    evaluations_traction_x = linear_elasticity_eq.make_nodes()[9].evaluate(
        {
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "normal_z": paddle.to_tensor(data=normal_z, dtype="float32"),
            "sigma_xx": paddle.to_tensor(data=sigma_xx, dtype="float32"),
            "sigma_xy": paddle.to_tensor(data=sigma_xy, dtype="float32"),
            "sigma_xz": paddle.to_tensor(data=sigma_xz, dtype="float32"),
        }
    )
    evaluations_traction_y = linear_elasticity_eq.make_nodes()[10].evaluate(
        {
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "normal_z": paddle.to_tensor(data=normal_z, dtype="float32"),
            "sigma_yy": paddle.to_tensor(data=sigma_yy, dtype="float32"),
            "sigma_xy": paddle.to_tensor(data=sigma_xy, dtype="float32"),
            "sigma_yz": paddle.to_tensor(data=sigma_yz, dtype="float32"),
        }
    )
    evaluations_traction_z = linear_elasticity_eq.make_nodes()[11].evaluate(
        {
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "normal_z": paddle.to_tensor(data=normal_z, dtype="float32"),
            "sigma_zz": paddle.to_tensor(data=sigma_zz, dtype="float32"),
            "sigma_xz": paddle.to_tensor(data=sigma_xz, dtype="float32"),
            "sigma_yz": paddle.to_tensor(data=sigma_yz, dtype="float32"),
        }
    )
    evaluations_navier_x = linear_elasticity_eq.make_nodes()[12].evaluate(
        {
            "u__t__t": paddle.to_tensor(data=u__t__t, dtype="float32"),
            "u__x__x": paddle.to_tensor(data=u__x__x, dtype="float32"),
            "v__x__y": paddle.to_tensor(data=v__x__y, dtype="float32"),
            "w__x__z": paddle.to_tensor(data=w__x__z, dtype="float32"),
            "u__y__y": paddle.to_tensor(data=u__y__y, dtype="float32"),
            "u__z__z": paddle.to_tensor(data=u__z__z, dtype="float32"),
        }
    )
    evaluations_navier_y = linear_elasticity_eq.make_nodes()[13].evaluate(
        {
            "v__t__t": paddle.to_tensor(data=v__t__t, dtype="float32"),
            "u__x__y": paddle.to_tensor(data=u__x__y, dtype="float32"),
            "v__y__y": paddle.to_tensor(data=v__y__y, dtype="float32"),
            "w__y__z": paddle.to_tensor(data=w__y__z, dtype="float32"),
            "v__x__x": paddle.to_tensor(data=v__x__x, dtype="float32"),
            "v__z__z": paddle.to_tensor(data=v__z__z, dtype="float32"),
        }
    )
    evaluations_navier_z = linear_elasticity_eq.make_nodes()[14].evaluate(
        {
            "w__t__t": paddle.to_tensor(data=w__t__t, dtype="float32"),
            "u__x__z": paddle.to_tensor(data=u__x__z, dtype="float32"),
            "v__y__z": paddle.to_tensor(data=v__y__z, dtype="float32"),
            "w__x__x": paddle.to_tensor(data=w__x__x, dtype="float32"),
            "w__y__y": paddle.to_tensor(data=w__y__y, dtype="float32"),
            "w__z__z": paddle.to_tensor(data=w__z__z, dtype="float32"),
        }
    )
    stress_disp_xx_eval_pred = evaluations_stress_disp_xx["stress_disp_xx"].numpy()
    stress_disp_yy_eval_pred = evaluations_stress_disp_yy["stress_disp_yy"].numpy()
    stress_disp_zz_eval_pred = evaluations_stress_disp_zz["stress_disp_zz"].numpy()
    stress_disp_xy_eval_pred = evaluations_stress_disp_xy["stress_disp_xy"].numpy()
    stress_disp_xz_eval_pred = evaluations_stress_disp_xz["stress_disp_xz"].numpy()
    stress_disp_yz_eval_pred = evaluations_stress_disp_yz["stress_disp_yz"].numpy()
    equilibrium_x_eval_pred = evaluations_equilibrium_x["equilibrium_x"].numpy()
    equilibrium_y_eval_pred = evaluations_equilibrium_y["equilibrium_y"].numpy()
    equilibrium_z_eval_pred = evaluations_equilibrium_z["equilibrium_z"].numpy()
    traction_x_eval_pred = evaluations_traction_x["traction_x"].numpy()
    traction_y_eval_pred = evaluations_traction_y["traction_y"].numpy()
    traction_z_eval_pred = evaluations_traction_z["traction_z"].numpy()
    navier_x_eval_pred = evaluations_navier_x["navier_x"].numpy()
    navier_y_eval_pred = evaluations_navier_y["navier_y"].numpy()
    navier_z_eval_pred = evaluations_navier_z["navier_z"].numpy()
    assert np.allclose(stress_disp_xx_eval_pred, stress_disp_xx_true), "Test Failed!"
    assert np.allclose(stress_disp_yy_eval_pred, stress_disp_yy_true), "Test Failed!"
    assert np.allclose(stress_disp_zz_eval_pred, stress_disp_zz_true), "Test Failed!"
    assert np.allclose(stress_disp_xy_eval_pred, stress_disp_xy_true), "Test Failed!"
    assert np.allclose(stress_disp_xz_eval_pred, stress_disp_xz_true), "Test Failed!"
    assert np.allclose(stress_disp_yz_eval_pred, stress_disp_yz_true), "Test Failed!"
    assert np.allclose(equilibrium_x_eval_pred, equilibrium_x_true), "Test Failed!"
    assert np.allclose(equilibrium_y_eval_pred, equilibrium_y_true), "Test Failed!"
    assert np.allclose(equilibrium_z_eval_pred, equilibrium_z_true), "Test Failed!"
    assert np.allclose(traction_x_eval_pred, traction_x_true), "Test Failed!"
    assert np.allclose(traction_y_eval_pred, traction_y_true), "Test Failed!"
    assert np.allclose(traction_z_eval_pred, traction_z_true), "Test Failed!"
    assert np.allclose(navier_x_eval_pred, navier_x_true, rtol=0.001), "Test Failed!"
    assert np.allclose(navier_y_eval_pred, navier_y_true, rtol=0.001), "Test Failed!"
    assert np.allclose(navier_z_eval_pred, navier_z_true, rtol=0.001), "Test Failed!"


def test_linear_elasticity_plane_stress_equations():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    normal_x = np.random.rand(1024, 1)
    normal_y = np.random.rand(1024, 1)
    u = np.exp(2 * x + y + t)
    v = np.exp(x + 2 * y + t)
    sigma_xx = np.sin(x) * np.cos(y)
    sigma_yy = np.cos(x) * np.sin(y)
    sigma_xy = np.sin(x) * np.sin(y)
    u__t__t = 1 * np.exp(2 * x + y + t)
    v__t__t = 1 * np.exp(x + 2 * y + t)
    u__x = 2 * np.exp(2 * x + y + t)
    u__y = 1 * np.exp(2 * x + y + t)
    u__x__x = 2 * 2 * np.exp(2 * x + y + t)
    u__y__y = 1 * 1 * np.exp(2 * x + y + t)
    u__x__y = 1 * 2 * np.exp(2 * x + y + t)
    u__y__x = u__x__y
    v__x = 1 * np.exp(x + 2 * y + t)
    v__y = 2 * np.exp(x + 2 * y + t)
    v__x__x = 1 * 1 * np.exp(x + 2 * y + t)
    v__y__y = 2 * 2 * np.exp(x + 2 * y + t)
    v__x__y = 2 * 1 * np.exp(x + 2 * y + t)
    v__y__x = v__x__y
    sigma_xx__x = np.cos(x) * np.cos(y)
    sigma_yy__y = np.cos(x) * np.cos(y)
    sigma_xy__x = np.cos(x) * np.sin(y)
    sigma_xy__y = np.sin(x) * np.cos(y)
    E = 1.0
    nu = 0.1
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    rho = 10.0
    w_z = -lambda_ / (lambda_ + 2 * mu) * (u__x + v__y)
    stress_disp_xx_true = lambda_ * (u__x + v__y + w_z) + 2 * mu * u__x - sigma_xx
    stress_disp_yy_true = lambda_ * (u__x + v__y + w_z) + 2 * mu * v__y - sigma_yy
    stress_disp_xy_true = mu * (u__y + v__x) - sigma_xy
    equilibrium_x_true = rho * u__t__t - (sigma_xx__x + sigma_xy__y)
    equilibrium_y_true = rho * v__t__t - (sigma_xy__x + sigma_yy__y)
    traction_x_true = normal_x * sigma_xx + normal_y * sigma_xy
    traction_y_true = normal_x * sigma_xy + normal_y * sigma_yy
    linear_elasticity_eq = LinearElasticityPlaneStress(nu=nu, E=E, rho=rho, time=True)
    evaluations_stress_disp_xx = linear_elasticity_eq.make_nodes()[0].evaluate(
        {
            "u__x": paddle.to_tensor(data=u__x, dtype="float32"),
            "v__y": paddle.to_tensor(data=v__y, dtype="float32"),
            "sigma_xx": paddle.to_tensor(data=sigma_xx, dtype="float32"),
        }
    )
    evaluations_stress_disp_yy = linear_elasticity_eq.make_nodes()[1].evaluate(
        {
            "u__x": paddle.to_tensor(data=u__x, dtype="float32"),
            "v__y": paddle.to_tensor(data=v__y, dtype="float32"),
            "sigma_yy": paddle.to_tensor(data=sigma_yy, dtype="float32"),
        }
    )
    evaluations_stress_disp_xy = linear_elasticity_eq.make_nodes()[2].evaluate(
        {
            "u__y": paddle.to_tensor(data=u__y, dtype="float32"),
            "v__x": paddle.to_tensor(data=v__x, dtype="float32"),
            "sigma_xy": paddle.to_tensor(data=sigma_xy, dtype="float32"),
        }
    )
    evaluations_equilibrium_x = linear_elasticity_eq.make_nodes()[3].evaluate(
        {
            "u__t__t": paddle.to_tensor(data=u__t__t, dtype="float32"),
            "sigma_xx__x": paddle.to_tensor(data=sigma_xx__x, dtype="float32"),
            "sigma_xy__y": paddle.to_tensor(data=sigma_xy__y, dtype="float32"),
        }
    )
    evaluations_equilibrium_y = linear_elasticity_eq.make_nodes()[4].evaluate(
        {
            "v__t__t": paddle.to_tensor(data=v__t__t, dtype="float32"),
            "sigma_xy__x": paddle.to_tensor(data=sigma_xy__x, dtype="float32"),
            "sigma_yy__y": paddle.to_tensor(data=sigma_yy__y, dtype="float32"),
        }
    )
    evaluations_traction_x = linear_elasticity_eq.make_nodes()[5].evaluate(
        {
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "sigma_xx": paddle.to_tensor(data=sigma_xx, dtype="float32"),
            "sigma_xy": paddle.to_tensor(data=sigma_xy, dtype="float32"),
        }
    )
    evaluations_traction_y = linear_elasticity_eq.make_nodes()[6].evaluate(
        {
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "sigma_yy": paddle.to_tensor(data=sigma_yy, dtype="float32"),
            "sigma_xy": paddle.to_tensor(data=sigma_xy, dtype="float32"),
        }
    )
    stress_disp_xx_eval_pred = evaluations_stress_disp_xx["stress_disp_xx"].numpy()
    stress_disp_yy_eval_pred = evaluations_stress_disp_yy["stress_disp_yy"].numpy()
    stress_disp_xy_eval_pred = evaluations_stress_disp_xy["stress_disp_xy"].numpy()
    equilibrium_x_eval_pred = evaluations_equilibrium_x["equilibrium_x"].numpy()
    equilibrium_y_eval_pred = evaluations_equilibrium_y["equilibrium_y"].numpy()
    traction_x_eval_pred = evaluations_traction_x["traction_x"].numpy()
    traction_y_eval_pred = evaluations_traction_y["traction_y"].numpy()
    assert np.allclose(stress_disp_xx_eval_pred, stress_disp_xx_true), "Test Failed!"
    assert np.allclose(stress_disp_yy_eval_pred, stress_disp_yy_true), "Test Failed!"
    assert np.allclose(stress_disp_xy_eval_pred, stress_disp_xy_true), "Test Failed!"
    assert np.allclose(equilibrium_x_eval_pred, equilibrium_x_true), "Test Failed!"
    assert np.allclose(equilibrium_y_eval_pred, equilibrium_y_true), "Test Failed!"
    assert np.allclose(traction_x_eval_pred, traction_x_true), "Test Failed!"
    assert np.allclose(traction_y_eval_pred, traction_y_true), "Test Failed!"


if __name__ == "__main__":
    test_linear_elasticity_equations()
    test_linear_elasticity_plane_stress_equations()
