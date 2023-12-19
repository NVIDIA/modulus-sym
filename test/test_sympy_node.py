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
from modulus.sym.utils.sympy import SympyToTorch
import sympy


def test_sympy_node():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    expr = sympy.Max(sympy.sin(x), sympy.cos(y))
    x_np = np.random.random(10)
    y_np = np.random.random(10)
    expr_np = np.maximum(np.sin(x_np), np.cos(y_np))
    sn = SympyToTorch(expr, "node")
    device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
        "cuda", "gpu"
    )
    x_th = paddle.to_tensor(data=x_np, dtype="float32", place=device)
    y_th = paddle.to_tensor(data=y_np, dtype="float32", place=device)
    assert np.allclose(x_th.cpu().detach().numpy(), x_np)
    assert np.allclose(y_th.cpu().detach().numpy(), y_np)
    var = {"x": x_th, "y": y_th}
    expr_th = sn(var)
    expr_th_out = expr_th["node"].cpu().detach().numpy()
    assert np.allclose(expr_th_out, expr_np, rtol=0.001), "SymPy printer test failed!"


if __name__ == "__main__":
    test_sympy_node()
