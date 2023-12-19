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
from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion


def test_advection_diffusion():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    T = np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u = np.exp(2 * x + y + z)
    v = np.exp(x + 2 * y + z)
    w = np.exp(x + y + 2 * z)
    rho = 1.0
    D = 0.1
    T__t = -np.sin(x) * np.sin(y) * np.sin(z) * np.sin(t)
    T__x = np.cos(x) * np.sin(y) * np.sin(z) * np.cos(t)
    T__y = np.sin(x) * np.cos(y) * np.sin(z) * np.cos(t)
    T__z = np.sin(x) * np.sin(y) * np.cos(z) * np.cos(t)
    T__x__x = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    T__y__y = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    T__z__z = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    advection = u * T__x + v * T__y + w * T__z
    diffusion = D * T__x__x + D * T__y__y + D * T__z__z
    curl = 0
    advection_diffusion_equation_true = T__t + advection + T * curl - diffusion
    eq = AdvectionDiffusion(T="T", D=D, rho=float(rho), dim=3, time=True)
    evaluations = eq.make_nodes()[0].evaluate(
        {
            "T__t": paddle.to_tensor(data=T__t, dtype="float32"),
            "T__x": paddle.to_tensor(data=T__x, dtype="float32"),
            "T__y": paddle.to_tensor(data=T__y, dtype="float32"),
            "T__z": paddle.to_tensor(data=T__z, dtype="float32"),
            "T__x__x": paddle.to_tensor(data=T__x__x, dtype="float32"),
            "T__y__y": paddle.to_tensor(data=T__y__y, dtype="float32"),
            "T__z__z": paddle.to_tensor(data=T__z__z, dtype="float32"),
            "u": paddle.to_tensor(data=u, dtype="float32"),
            "v": paddle.to_tensor(data=v, dtype="float32"),
            "w": paddle.to_tensor(data=w, dtype="float32"),
        }
    )
    eq_eval = evaluations["advection_diffusion_T"].numpy()
    assert np.allclose(eq_eval, advection_diffusion_equation_true), "Test Failed!"


if __name__ == "__main__":
    test_advection_diffusion()
