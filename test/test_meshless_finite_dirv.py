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
from modulus.sym.eq.derivatives import MeshlessFiniteDerivative
from modulus.sym.node import Node
from modulus.sym.key import Key
from modulus.sym.graph import Graph


class SineNet(paddle.nn.Layer):
    def forward(self, inputs):
        return {
            "y": inputs["w"] ** 3 * paddle.sin(x=inputs["x"]),
            "z": inputs["w"] * paddle.cos(x=inputs["x"]),
        }


class ParabolaNet(paddle.nn.Layer):
    def forward(self, inputs):
        return {"p": inputs["nu"] ** 3 + inputs["x"], "q": 2 * inputs["z"]}


def test_meshless_finite_deriv():
    function_node = Node(
        inputs=[Key("w"), Key("x")],
        outputs=[Key("y"), Key("z")],
        evaluate=SineNet(),
        name="Test Node",
    )
    deriv = MeshlessFiniteDerivative.make_node(
        node_model=function_node,
        derivatives=[
            Key("y", derivatives=[Key("x"), Key("w")]),
            Key("y", derivatives=[Key("x")]),
            Key("y", derivatives=[Key("w"), Key("w"), Key("w")]),
            Key("z", derivatives=[Key("x"), Key("x")]),
            Key("z", derivatives=[Key("x"), Key("x"), Key("x"), Key("x")]),
        ],
        dx=0.01,
        order=2,
        max_batch_size=15,
    )
    inputs = {
        "x": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
        "w": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
    }
    inputs.update(function_node.evaluate(inputs))
    outputs = deriv.evaluate(inputs)
    assert paddle.allclose(
        x=outputs["y__x"].astype(dtype="float64"),
        y=inputs["w"] ** 3 * paddle.cos(x=inputs["x"]),
        atol=0.001,
    ).item(), "First derivative test failed"
    assert paddle.allclose(
        x=outputs["z__x__x"].astype(dtype="float64"),
        y=-inputs["w"] * paddle.cos(x=inputs["x"]),
        atol=0.001,
    ).item(), "Second derivative test failed"
    assert paddle.allclose(
        x=outputs["y__x__w"].astype(dtype="float64"),
        y=3 * inputs["w"] ** 2 * paddle.cos(x=inputs["x"]),
        atol=0.001,
    ).item(), "Mixed second derivative test failed"
    assert paddle.allclose(
        x=outputs["y__w__w__w"].astype(dtype="float64"),
        y=6 * paddle.sin(x=inputs["x"]),
        atol=0.001,
    ).item(), "Third derivative test failed"
    assert paddle.allclose(
        x=outputs["z__x__x__x__x"].astype(dtype="float64"),
        y=inputs["w"] * paddle.cos(x=inputs["x"]),
        atol=0.001,
    ).item(), "Forth derivative test failed"
    deriv = MeshlessFiniteDerivative.make_node(
        node_model=function_node,
        derivatives=[
            Key("y", derivatives=[Key("x")]),
            Key("z", derivatives=[Key("x"), Key("x")]),
        ],
        dx=0.01,
        order=4,
        max_batch_size=20,
    )
    inputs = {
        "x": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
        "w": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
    }
    inputs.update(function_node.evaluate(inputs))
    outputs = deriv.evaluate(inputs)
    assert paddle.allclose(
        x=outputs["y__x"].astype(dtype="float64"),
        y=inputs["w"] ** 3 * paddle.cos(x=inputs["x"]),
        atol=0.01,
    ).item(), "Forth order first derivative test failed"
    assert paddle.allclose(
        x=outputs["z__x__x"].astype(dtype="float64"),
        y=-inputs["w"] * paddle.cos(x=inputs["x"]),
        atol=0.01,
    ).item(), "Forth order second derivative test failed"
    function_node_2 = Node(
        inputs=[Key("nu"), Key("w"), Key("z")],
        outputs=[Key("p"), Key("q")],
        evaluate=ParabolaNet(),
        name="Test Node 2",
    )
    deriv = MeshlessFiniteDerivative.make_node(
        node_model=Graph(
            nodes=[function_node, function_node_2],
            invar=[Key("w"), Key("x"), Key("nu")],
            req_names=[Key("p"), Key("q")],
        ),
        derivatives=[
            Key("p", derivatives=[Key("nu")]),
            Key("q", derivatives=[Key("x"), Key("w")]),
        ],
        dx=0.01,
    )
    inputs = {
        "x": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
        "w": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
        "nu": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
    }
    outputs = deriv.evaluate(inputs)
    assert paddle.allclose(
        x=outputs["p__nu"].astype(dtype="float64"), y=3 * inputs["nu"] ** 2, atol=0.001
    ).item(), "Multi-node first derivative test failed"
    assert paddle.allclose(
        x=outputs["q__x__w"].astype(dtype="float64"),
        y=2 * -paddle.sin(x=inputs["x"]),
        atol=0.001,
    ).item(), "Multi-node second derivative test failed"

    def dx_func(count: int):
        if count == 1:
            return 10.0
        else:
            return 0.01

    deriv = MeshlessFiniteDerivative.make_node(
        node_model=function_node,
        derivatives=[Key("y", derivatives=[Key("x")])],
        dx=dx_func,
        order=2,
    )
    inputs = {
        "x": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
        "w": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
    }
    inputs.update(function_node.evaluate(inputs))
    outputs_1 = deriv.evaluate(inputs)
    outputs_2 = deriv.evaluate(inputs)
    assert not paddle.allclose(
        x=outputs_1["y__x"].astype(dtype="float64"),
        y=inputs["w"] ** 3 * paddle.cos(x=inputs["x"]),
        atol=0.001,
    ).item(), "Callable dx first derivative test failed"
    assert paddle.allclose(
        x=outputs_2["y__x"].astype(dtype="float64"),
        y=inputs["w"] ** 3 * paddle.cos(x=inputs["x"]),
        atol=0.001,
    ).item(), "Callable dx first derivative test failed"


class GradModel(paddle.nn.Layer):
    def forward(self, inputs):
        return {"u": paddle.cos(x=inputs["x"]), "v": paddle.sin(x=inputs["y"])}


def test_meshless_finite_deriv_grads():
    model = GradModel()
    dx = 0.01
    deriv = MeshlessFiniteDerivative.make_node(
        node_model=model,
        derivatives=[
            Key("u", derivatives=[Key("x")]),
            Key("v", derivatives=[Key("y"), Key("y")]),
        ],
        dx=dx,
    )
    inputs_mfd = {
        "x": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
        "y": paddle.randn(shape=[5, 1]).astype(dtype="float64"),
    }
    inputs_mfd["x"].stop_gradient = not True
    inputs_mfd["y"].stop_gradient = not True
    inputs_mfd.update(model.forward(inputs_mfd))
    outputs = deriv.evaluate(inputs_mfd)
    loss = outputs["u__x"].sum()
    loss.backward()
    inputs_auto = inputs_mfd["x"].detach().clone()
    inputs_auto.stop_gradient = not True
    inputs_up1 = paddle.cos(x=inputs_auto + dx)
    inputs_um1 = paddle.cos(x=inputs_auto - dx)
    grad = (inputs_up1 - inputs_um1) / (2.0 * dx)
    loss = grad.sum()
    loss.backward()
    assert paddle.allclose(
        x=inputs_auto.grad, y=inputs_mfd["x"].grad, atol=0.001
    ).item(), "First derivative gradient test failed"
    loss = outputs["v__y__y"].sum()
    loss.backward()
    inputs_auto = inputs_mfd["y"].detach().clone()
    inputs_auto.stop_gradient = not True
    inputs = paddle.sin(x=inputs_auto)
    inputs_up1 = paddle.sin(x=inputs_auto + dx)
    inputs_um1 = paddle.sin(x=inputs_auto - dx)
    grad = (inputs_up1 - 2 * inputs + inputs_um1) / (dx * dx)
    loss = grad.sum()
    loss.backward()
    assert paddle.allclose(
        x=inputs_auto.grad, y=inputs_mfd["y"].grad, atol=0.001
    ).item(), "Second derivative gradient test failed"


if __name__ == "__main__":
    test_meshless_finite_deriv()
    test_meshless_finite_deriv_grads()
