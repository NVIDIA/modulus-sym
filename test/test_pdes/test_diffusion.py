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
from modulus.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface


def test_diffusion_equation():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    u = np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    D = 0.1
    Q = 0.1
    u__t = -np.sin(x) * np.sin(y) * np.sin(z) * np.sin(t)
    u__x = np.cos(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u__y = np.sin(x) * np.cos(y) * np.sin(z) * np.cos(t)
    u__z = np.sin(x) * np.sin(y) * np.cos(z) * np.cos(t)
    u__x__x = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u__y__y = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u__z__z = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    diffusion_equation_true = u__t - D * u__x__x - D * u__y__y - D * u__z__z - Q
    eq = Diffusion(T="u", D=D, Q=Q, dim=3, time=True)
    evaluations = eq.make_nodes()[0].evaluate(
        {
            "u__x__x": paddle.to_tensor(data=u__x__x, dtype="float32"),
            "u__y__y": paddle.to_tensor(data=u__y__y, dtype="float32"),
            "u__z__z": paddle.to_tensor(data=u__z__z, dtype="float32"),
            "u__t": paddle.to_tensor(data=u__t, dtype="float32"),
        }
    )
    eq_eval = evaluations["diffusion_u"].numpy()
    assert np.allclose(eq_eval, diffusion_equation_true), "Test Failed!"


def test_diffusion_interface():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    normal_x = np.random.rand(1024, 1)
    normal_y = np.random.rand(1024, 1)
    normal_z = np.random.rand(1024, 1)
    u_1 = np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u_2 = np.cos(x) * np.cos(y) * np.cos(z) * np.sin(t)
    D_1 = 0.1
    D_2 = 100
    u_1__x = np.cos(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u_1__y = np.sin(x) * np.cos(y) * np.sin(z) * np.cos(t)
    u_1__z = np.sin(x) * np.sin(y) * np.cos(z) * np.cos(t)
    u_2__x = -np.sin(x) * np.cos(y) * np.cos(z) * np.sin(t)
    u_2__y = -np.cos(x) * np.sin(y) * np.cos(z) * np.sin(t)
    u_2__z = -np.cos(x) * np.cos(y) * np.sin(z) * np.sin(t)
    diffusion_interface_dirichlet_u_1_u_2_true = u_1 - u_2
    diffusion_interface_neumann_u_1_u_2_true = D_1 * (
        normal_x * u_1__x + normal_y * u_1__y + normal_z * u_1__z
    ) - D_2 * (normal_x * u_2__x + normal_y * u_2__y + normal_z * u_2__z)
    eq = DiffusionInterface(T_1="u_1", T_2="u_2", D_1=D_1, D_2=D_2, dim=3, time=True)
    evaluations = eq.make_nodes()[0].evaluate(
        {
            "u_1": paddle.to_tensor(data=u_1, dtype="float32"),
            "u_2": paddle.to_tensor(data=u_2, dtype="float32"),
        }
    )
    eq_1_eval = evaluations["diffusion_interface_dirichlet_u_1_u_2"].numpy()
    evaluations = eq.make_nodes()[1].evaluate(
        {
            "u_1": paddle.to_tensor(data=u_1, dtype="float32"),
            "u_2": paddle.to_tensor(data=u_2, dtype="float32"),
            "u_1__x": paddle.to_tensor(data=u_1__x, dtype="float32"),
            "u_1__y": paddle.to_tensor(data=u_1__y, dtype="float32"),
            "u_1__z": paddle.to_tensor(data=u_1__z, dtype="float32"),
            "u_2__x": paddle.to_tensor(data=u_2__x, dtype="float32"),
            "u_2__y": paddle.to_tensor(data=u_2__y, dtype="float32"),
            "u_2__z": paddle.to_tensor(data=u_2__z, dtype="float32"),
            "normal_x": paddle.to_tensor(data=normal_x, dtype="float32"),
            "normal_y": paddle.to_tensor(data=normal_y, dtype="float32"),
            "normal_z": paddle.to_tensor(data=normal_z, dtype="float32"),
        }
    )
    eq_2_eval = evaluations["diffusion_interface_neumann_u_1_u_2"].numpy()
    assert np.allclose(
        eq_1_eval, diffusion_interface_dirichlet_u_1_u_2_true
    ), "Test Failed!"
    assert np.allclose(
        eq_2_eval, diffusion_interface_neumann_u_1_u_2_true
    ), "Test Failed!"


if __name__ == "__main__":
    test_diffusion_equation()
    test_diffusion_interface()
