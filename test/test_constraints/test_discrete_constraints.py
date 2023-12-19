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
from sympy import Symbol
from modulus.sym.node import Node
from modulus.sym.domain.constraint.discrete import (
    SupervisedGridConstraint,
    DeepONetConstraint_Data,
    DeepONetConstraint_Physics,
)
from modulus.sym.dataset import DictGridDataset


def test_SupervisedGridConstraint():
    """define a parabola node, create grid constraint over it and check its loss is zero"""
    node = Node.from_sympy(Symbol("x") ** 2 + Symbol("y") ** 2, "u")
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    u = x**2 + y**2
    dataset = DictGridDataset(
        invar={"x": x[np.newaxis, :], "y": y[np.newaxis, :]},
        outvar={"u": u[np.newaxis, :]},
    )
    constraint = SupervisedGridConstraint(nodes=[node], dataset=dataset, batch_size=1)
    constraint.load_data()
    constraint.forward()
    loss = constraint.loss(step=0)
    assert paddle.isclose(
        x=loss["u"], y=paddle.to_tensor(data=0.0), rtol=1e-05, atol=1e-05
    )


def test_DeepONetConstraints():
    """define a parabola node, create deeponet constraints over it and check their losses are zero"""
    node = Node.from_sympy(Symbol("x") ** 2 + Symbol("y") ** 2, "u")
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    u = x**2 + y**2
    invar_branch = {"x": x[np.newaxis, :]}
    invar_trunk = {"y": y[np.newaxis, :]}
    outvar = {"u": u[np.newaxis, :]}
    constraint = DeepONetConstraint_Data(
        nodes=[node],
        invar_branch=invar_branch,
        invar_trunk=invar_trunk,
        outvar=outvar,
        batch_size=1,
    )
    constraint.load_data()
    constraint.forward()
    loss = constraint.loss(step=0)
    assert paddle.isclose(
        x=loss["u"], y=paddle.to_tensor(data=0.0), rtol=1e-05, atol=1e-05
    )

    class Parabola(paddle.nn.Layer):
        def forward(self, invar):
            x, y = invar["x"], invar["y"]
            u = x**2 + y**2
            u = u.reshape((-1, 1))
            return {"u": u}

    node = Node(inputs=["x", "y"], outputs="u", evaluate=Parabola())
    constraint = DeepONetConstraint_Physics(
        nodes=[node],
        invar_branch=invar_branch,
        invar_trunk=invar_trunk,
        outvar=outvar,
        batch_size=1,
    )
    constraint.load_data()
    constraint.forward()
    loss = constraint.loss(step=0)
    assert paddle.isclose(
        x=loss["u"], y=paddle.to_tensor(data=0.0), rtol=1e-05, atol=1e-05
    )


if __name__ == "__main__":
    test_SupervisedGridConstraint()
    test_DeepONetConstraints()
