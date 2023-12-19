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

import sys
import paddle
import time
from typing import Dict, List, Optional
from modulus.sym.key import Key
from modulus.sym.constants import diff
from modulus.sym.node import Node
from modulus.sym.graph import Graph
from modulus.sym.eq.derivatives import MeshlessFiniteDerivative


class Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        x, y, z = inputs["x"], inputs["y"], inputs["z"]
        return {
            "u": 1.5 * x * x + paddle.sin(x=y) + paddle.exp(x=z),
            "v": 2 * x * x + paddle.cos(x=y) + paddle.exp(x=-z),
            "w": 1.5 * x * x + paddle.sin(x=y) + paddle.exp(x=z),
            "p": 2 * x * x + paddle.cos(x=y) + paddle.exp(x=-z),
        }


class Loss(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.input_keys: List[str] = [diff("u", "x"), diff("v", "y"), diff("w", "z")]

    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        divergence = (
            inputs[self.input_keys[0]]
            + inputs[self.input_keys[1]]
            + inputs[self.input_keys[2]]
        )
        return {"divergence_loss": paddle.square(x=divergence).mean()}


def validate_divergence_loss(x, y, z, divergence_loss, rtol=1e-05, atol=1e-08):
    dudx = 3 * x
    dvdy = -paddle.sin(x=y)
    dwdz = paddle.exp(x=z)
    divergence_loss_exact = paddle.square(x=dudx + dvdy + dwdz).mean()
    assert paddle.allclose(
        x=divergence_loss, y=divergence_loss_exact, rtol=rtol, atol=atol
    ).item()


def test_graph():
    device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
        "cuda", "gpu"
    )
    batch_size = 128
    out_0 = paddle.rand(shape=[batch_size, 1], dtype="float32")
    out_0.stop_gradient = not True
    x = out_0.to(device)
    out_1 = paddle.rand(shape=[batch_size, 1], dtype="float32")
    out_1.stop_gradient = not True
    y = out_1.to(device)
    out_2 = paddle.rand(shape=[batch_size, 1], dtype="float32")
    out_2.stop_gradient = not True
    z = out_2.to(device)
    model = Model()
    model_node = Node(["x", "y", "z"], ["u", "v", "w", "p"], model, name="Model")
    loss = Loss()
    loss_node = Node(
        [diff("u", "x"), diff("v", "y"), diff("w", "z")],
        ["divergence_loss"],
        loss,
        name="Loss",
    )
    nodes = [model_node, loss_node]
    input_vars = [Key.from_str("x"), Key.from_str("y"), Key.from_str("z")]
    output_vars = [
        Key.from_str("u"),
        Key.from_str("v"),
        Key.from_str("w"),
        Key.from_str("p"),
        Key.from_str("divergence_loss"),
    ]
    graph = Graph(nodes, input_vars, output_vars)
    input_dict = dict(zip((str(v) for v in input_vars), [x, y, z]))
    output_dict = graph(input_dict)
    validate_divergence_loss(x, y, z, output_dict["divergence_loss"])


def test_graph_no_loss_node():
    device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
        "cuda", "gpu"
    )
    batch_size = 128
    out_3 = paddle.rand(shape=[batch_size, 1], dtype="float32")
    out_3.stop_gradient = not True
    x = out_3.to(device)
    out_4 = paddle.rand(shape=[batch_size, 1], dtype="float32")
    out_4.stop_gradient = not True
    y = out_4.to(device)
    out_5 = paddle.rand(shape=[batch_size, 1], dtype="float32")
    out_5.stop_gradient = not True
    z = out_5.to(device)
    model = Model()
    model_node = Node(["x", "y", "z"], ["u", "v", "w", "p"], model, name="Model")
    loss = Loss()
    loss_node = Node(
        [diff("u", "x"), diff("v", "y"), diff("w", "z")],
        ["divergence_loss"],
        loss,
        name="Loss",
    )
    nodes = [model_node]
    input_vars = [Key.from_str("x"), Key.from_str("y"), Key.from_str("z")]
    output_vars = [Key.from_str("u__x"), Key.from_str("v__y"), Key.from_str("w__z")]
    graph = Graph(nodes, input_vars, output_vars)
    input_dict = dict(zip((str(v) for v in input_vars), [x, y, z]))
    output_dict = graph(input_dict)
    loss = Loss()
    output_dict.update(loss(output_dict))
    validate_divergence_loss(x, y, z, output_dict["divergence_loss"])


def test_mfd_graph():
    device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
        "cuda", "gpu"
    )
    batch_size = 32
    out_6 = paddle.rand(shape=[batch_size, 1], dtype="float32")
    out_6.stop_gradient = not True
    x = out_6.to(device)
    out_7 = paddle.rand(shape=[batch_size, 1], dtype="float32")
    out_7.stop_gradient = not True
    y = out_7.to(device)
    out_8 = paddle.rand(shape=[batch_size, 1], dtype="float32")
    out_8.stop_gradient = not True
    z = out_8.to(device)
    model = Model()
    model_node = Node(["x", "y", "z"], ["u", "v", "w", "p"], model, name="Model")
    loss = Loss()
    loss_node = Node(
        [diff("u", "x"), diff("v", "y"), diff("w", "z")],
        ["divergence_loss"],
        loss,
        name="Loss",
    )
    nodes = [model_node, loss_node]
    input_vars = [Key.from_str("x"), Key.from_str("y"), Key.from_str("z")]
    output_vars = [
        Key.from_str("u"),
        Key.from_str("v"),
        Key.from_str("w"),
        Key.from_str("p"),
        Key.from_str("divergence_loss"),
    ]
    mfd_node = MeshlessFiniteDerivative.make_node(
        node_model=model,
        derivatives=[
            Key("u", derivatives=[Key("x")]),
            Key("v", derivatives=[Key("y")]),
            Key("w", derivatives=[Key("z")]),
        ],
        dx=0.001,
    )
    graph = Graph(nodes + [mfd_node], input_vars, output_vars)
    input_dict = dict(zip((str(v) for v in input_vars), [x, y, z]))
    output_dict = graph(input_dict)
    validate_divergence_loss(x, y, z, output_dict["divergence_loss"], atol=0.001)


if __name__ == "__main__":
    test_graph()
    test_graph_no_loss_node()
    test_mfd_graph()
