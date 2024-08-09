# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import copy
import torch
import torch.nn as nn
import numpy as np
import logging

from dataclasses import dataclass, field
import modulus
from modulus.sym.key import Key
from modulus.sym.graph import Graph
from modulus.sym.eq.pdes.navier_stokes import NavierStokes

from typing import Dict, List, Set, Optional, Union, Callable
from modulus.sym.node import Node

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from modulus.sym.eq.spatial_grads.spatial_grads import (
    GradientCalculator,
    compute_connectivity_tensor,
    compute_stencil,
)


def plot_fields(field, name):
    fig, axs = plt.subplots(1, 3)
    for i, ax in enumerate(axs):
        if i < 2:
            im = ax.imshow(field.detach().cpu().numpy()[:, :, field.shape[2] // 2])
        else:
            im = ax.imshow(field.detach().cpu().numpy()[field.shape[2] // 2, :, :])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class ModelMetaData:
    """Data class for storing essential meta data needed for all Modulus Models"""

    # Model info
    name: str = "ModulusModule"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = False
    amp_cpu: bool = None
    amp_gpu: bool = None
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    onnx_gpu: bool = None
    onnx_cpu: bool = None
    onnx_runtime: bool = False
    trt: bool = False
    # Physics informed
    var_dim: int = -1
    func_torch: bool = False
    auto_grad: bool = False
    output_types: List[str] = field(
        default_factory=list
    )  # options available "point_cloud", "grid" and "mesh_graph"
    preferred_grad_method: str = None  # options availabe "autodiff", "meshless_finite_difference", "finite_difference", "spectral", "least_squares"

    def __post_init__(self):
        self.amp_cpu = self.amp if self.amp_cpu is None else self.amp_cpu
        self.amp_gpu = self.amp if self.amp_gpu is None else self.amp_gpu
        self.onnx_cpu = self.onnx if self.onnx_cpu is None else self.onnx_cpu
        self.onnx_gpu = self.onnx if self.onnx_gpu is None else self.onnx_gpu


@dataclass
class UNetMetaData(ModelMetaData):
    name: str = "UNet"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True

    # Physics Informed
    # preferred_grad_method: str = "finite_difference"    # options availabe "autodiff", "meshless_finite_difference", "finite_difference", "spectral", "least_squares"
    preferred_grad_method: str = "spectral"

    def __post_init__(self):
        self.output_types = ["grid"]


class UNet(modulus.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        output_types=None,
        preferred_grad_method=None,
    ):
        meta = UNetMetaData()
        if output_types is not None:
            meta.output_types = output_types
        if preferred_grad_method is not None:
            meta.preferred_grad_method = preferred_grad_method
        super(UNet, self).__init__(meta=meta)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # compute u, v, w, p using trignometric functions
        u = (
            torch.sin(1 * x[:, 0:1])
            + torch.sin(8 * x[:, 1:2])
            + torch.sin(4 * x[:, 2:3])
        )
        v = (
            torch.sin(8 * x[:, 0:1])
            + torch.sin(2 * x[:, 1:2])
            + torch.sin(1 * x[:, 2:3])
        )
        w = (
            torch.sin(2 * x[:, 0:1])
            + torch.sin(2 * x[:, 1:2])
            + torch.sin(9 * x[:, 2:3])
        )
        p = (
            torch.sin(1 * x[:, 0:1])
            + torch.sin(1 * x[:, 1:2])
            + torch.sin(1 * x[:, 2:3])
        )

        return torch.cat([u, v, w, p], dim=1)


class PhysicsInformer(object):
    def __init__(
        self,
        required_outputs,
        equations,
        grad_method,
        available_inputs=None,
        fd_dx=0.001,  # only applies for FD and Meshless FD. Ignored for the rest
        bounds=[2 * np.pi, 2 * np.pi, 2 * np.pi],
    ):
        super().__init__()
        self.available_inputs = available_inputs
        self.required_outputs = required_outputs
        self.equations = equations
        self.dim = equations.dim
        self.grad_method = grad_method
        self.fd_dx = fd_dx
        self.bounds = bounds
        self.grad_calc = GradientCalculator()
        self.nodes = self.equations.make_nodes()

        self.required_inputs = self._find_required_inputs()
        self.graph = self._create_graph()

    def _find_required_inputs(self):
        node_outputs = [str(n.outputs[0]) for n in self.nodes]
        node_inputs = set()

        for node in self.required_outputs:
            if node not in node_outputs:
                raise ValueError(
                    f"{node} does not appear in the equation outputs provided. "
                    + f"Please choose from {node_outputs}"
                )

        fd, sd, others = self._extract_derivatives()

        for input in fd | sd | others:
            node_inputs.add(input)

        for node in self.nodes:
            if str(node.outputs[0]) in self.required_outputs and node.inputs:
                node_inputs.update(map(str, node.inputs))

        node_inputs = list(node_inputs)

        if self.grad_method == "meshless_finite_difference":
            node_inputs = self._expand_for_meshless_fd(node_inputs)
        elif self.grad_method == "autodiff":
            node_inputs.append("coordinates")
        elif self.grad_method == "least_squares":
            node_inputs.extend(["coordinates", "nodes", "edges"])

        # print(f"To compute the required {self.required_outputs}, using {self.grad_method} method, {node_inputs} will be required. Please provide them during the forward call")
        return node_inputs

    def _expand_for_meshless_fd(self, node_inputs):
        node_inputs_new = copy.deepcopy(node_inputs)
        for node in node_inputs:
            node_inputs_new.extend(
                [
                    f"{node}>>x::1",
                    f"{node}>>x::-1",
                    f"{node}>>y::1",
                    f"{node}>>y::-1",
                    f"{node}>>z::1",
                    f"{node}>>z::-1",
                ]
            )
        return node_inputs_new

    def _create_graph(self):
        first_deriv, second_deriv, _ = self._extract_derivatives()

        input_keys_sym = [Key(k) for k in self.required_inputs]
        output_keys_sym = [Key(k) for k in self.required_outputs]

        diff_nodes = self._create_diff_nodes(first_deriv, dim=self.dim, order=1)
        diff_nodes += self._create_diff_nodes(second_deriv, dim=self.dim, order=2)

        return Graph(self.nodes, input_keys_sym, output_keys_sym, diff_nodes=diff_nodes)

    def _extract_derivatives(self):
        first_deriv, second_deriv, other_derivs = set(), set(), set()

        for node in self.nodes:
            if str(node.outputs[0]) in self.required_outputs:
                for derr in node.derivatives:
                    self._process_derivative(
                        derr, first_deriv, second_deriv, other_derivs
                    )

        first_deriv_consolidated = {i.split("__")[0] for i in first_deriv}
        second_deriv_consolidated = {i.split("__")[0] for i in second_deriv}

        return first_deriv_consolidated, second_deriv_consolidated, other_derivs

    def _process_derivative(self, derr, first_deriv, second_deriv, other_derivs):
        if str(derr).count("__") > 2:
            raise ValueError("Only second order PDEs are supported presently")

        allowed_derr_vars = ["x", "y", "z"]
        for var in str(derr).split("__")[1:]:
            if var not in allowed_derr_vars:
                logging.warning(
                    f"Detected derivative w.r.t {var}. "
                    + f"Note, derivatives w.r.t only {allowed_derr_vars} vars are "
                    + f"computed automatically. The {str(derr)} will have to be "
                    + "provided as an input during the forward call."
                )
                other_derivs.add(str(derr))

        if (
            str(derr).count("__") == 2
            and str(derr).split("__")[1] != str(derr).split("__")[2]
        ):
            raise ValueError(
                f"Found {str(derr)}. PDEs with Mixed Derivatives not supported presently"
            )

        if str(derr).count("__") == 1:
            first_deriv.add(str(derr))
        elif str(derr).count("__") == 2:
            second_deriv.add(str(derr))

    def _create_diff_nodes(self, derivatives, dim, order):
        diff_nodes = []
        for derr in derivatives:
            node = self._create_diff_node(derr, dim, order)
            if node:
                diff_nodes.append(node)
        return diff_nodes

    def _create_diff_node(self, derr, dim, order):
        methods = {
            "finite_difference": self._fd_gradient_module,
            "spectral": self._spectral_gradient_module,
            "least_squares": self._ls_gradient_module,
            "autodiff": self._autodiff_gradient_module,
            "meshless_finite_difference": self._meshless_fd_gradient_module,
        }

        if self.grad_method in methods:
            return Node(
                str(derr).split("__")[0],
                self._derivative_keys(derr, order),
                methods[self.grad_method](derr, dim, order),
            )

    def _derivative_keys(self, derr, order):
        base_keys = ["__x", "__y", "__z"]
        return [f"{derr}{k * order}" for k in base_keys]

    def _fd_gradient_module(self, derr, dim, order):
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            str(derr).split("__")[0],
            dx=self.fd_dx,
            dim=dim,
            order=order,
        )

    def _spectral_gradient_module(self, derr, dim, order):
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            str(derr).split("__")[0],
            ell=self.bounds,
            dim=dim,
            order=order,
        )

    def _ls_gradient_module(self, derr, dim, order):
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            str(derr).split("__")[0],
            dim=dim,
            order=order,
        )

    def _autodiff_gradient_module(self, derr, dim, order):
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            str(derr).split("__")[0],
            dim=dim,
            order=order,
        )

    def _meshless_fd_gradient_module(self, derr, dim, order):
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            str(derr).split("__")[0],
            dx=self.fd_dx,
            dim=dim,
            order=order,
        )

    def forward(self, inputs):
        if self.grad_method == "least_squares":
            connectivity_tensor = compute_connectivity_tensor(
                inputs["coordinates"], inputs["nodes"], inputs["edges"]
            )
            inputs["connectivity_tensor"] = connectivity_tensor

        return self.graph.forward(inputs)


if __name__ == "__main__":

    # Prepare reference data
    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True)
    y = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True)
    z = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)
    coords_unstructured = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    # Edge information
    edges = []
    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                index = i * steps * steps + j * steps + k
                # Check and add connections in x direction
                if i < steps - 1:
                    edges.append([index, index + steps * steps])
                # Check and add connections in y direction
                if j < steps - 1:
                    edges.append([index, index + steps])
                # Check and add connections in z direction
                if k < steps - 1:
                    edges.append([index, index + 1])

    # Convert edges to tensor
    edges = torch.tensor(edges)
    node_ids = torch.arange(coords_unstructured.size(0)).reshape(-1, 1)

    u = (
        torch.sin(1 * coords[:, 0:1])
        + torch.sin(8 * coords[:, 1:2])
        + torch.sin(4 * coords[:, 2:3])
    )
    v = (
        torch.sin(8 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(1 * coords[:, 2:3])
    )
    w = (
        torch.sin(2 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(9 * coords[:, 2:3])
    )
    p = (
        torch.sin(1 * coords[:, 0:1])
        + torch.sin(1 * coords[:, 1:2])
        + torch.sin(1 * coords[:, 2:3])
    )

    p__x = torch.cos(1 * coords[:, 0:1])
    u__x = torch.cos(1 * coords[:, 0:1])
    u__y = 8 * torch.cos(8 * coords[:, 1:2])
    u__z = 4 * torch.cos(4 * coords[:, 2:3])
    v__y = 2 * torch.cos(2 * coords[:, 1:2])
    w__z = 9 * torch.cos(9 * coords[:, 2:3])
    u__x__x = -1 * torch.sin(1 * coords[:, 0:1])
    u__y__y = -64 * torch.sin(8 * coords[:, 1:2])
    u__z__z = -16 * torch.sin(4 * coords[:, 2:3])

    true_cont = u__x + v__y + w__z
    true_mom_x = (
        u * u__x
        + v * u__y
        + w * u__z
        + p__x
        - 0.01 * u__x__x
        - 0.01 * u__y__y
        - 0.01 * u__z__z
    )

    print("Analytical: ", true_mom_x.shape)
    plot_fields(true_mom_x[0, 0], "true_momentum_x")

    # setup the physics informer
    # Required args: inputs, required_outputs, model, equations
    # Optional args: grad_method - default it will try to choose based on model
    # type / metadata but can be overriden as long as it fits in the bounds of what is possible.
    # the object then figures out what function to use when called "compute"
    # dimensionality information to be inferred from the equation class, make it an optional arg.
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)

    # Debugging PhysicsInformer
    # model = UNet(output_types=["grid"], preferred_grad_method="finite_difference")
    # phy_informer = PhysicsInformer(
    #     required_outputs=["momentum_x", "continuity", "momentum_y", "momentum_z"],
    #     equations=ns,
    #     grad_method="finite_difference",
    #     fd_dx=(2 * np.pi / steps)
    # )
    # pred_outvar = model(coords)
    # physics_loss_dict = phy_informer.forward(
    #     {
    #         "u": pred_outvar[:, 0:1],
    #         "u__t": pred_outvar[:, 0:1],    # compute temporal derivative on your own
    #         "v": pred_outvar[:, 1:2],
    #         "v__t": pred_outvar[:, 0:1],    # compute temporal derivative on your own
    #         "w": pred_outvar[:, 2:3],
    #         "w__t": pred_outvar[:, 0:1],    # compute temporal derivative on your own
    #         "p": pred_outvar[:, 3:4],
    #     },
    # )
    # phy_loss = physics_loss_dict["momentum_x"]
    # print(phy_loss.shape)
    # exit()
    # TEST Grid data, Finite Difference
    # Set the model meta data explicitly (for demo purposes only)
    model = UNet(output_types=["grid"], preferred_grad_method="finite_difference")
    phy_informer = PhysicsInformer(
        required_outputs=["momentum_x"],
        equations=ns,
        grad_method="finite_difference",
        fd_dx=(2 * np.pi / steps),  # computed based on the grid spacing
    )

    pred_outvar = model(coords)
    physics_loss_dict = phy_informer.forward(
        {
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )
    phy_loss = physics_loss_dict["momentum_x"]

    print("Finite Difference: ", phy_loss.shape)
    plot_fields(phy_loss[0, 0], "fd_momentum_x")

    # TEST Grid data, Finite Difference
    # Set the model meta data explicitly (for demo purposes only)
    model = UNet(output_types=["grid"], preferred_grad_method="spectral")

    phy_informer = PhysicsInformer(
        required_outputs=["momentum_x"],
        equations=ns,
        grad_method="spectral",
        bounds=[2 * np.pi, 2 * np.pi, 2 * np.pi],
    )

    pred_outvar = model(coords)
    physics_loss_dict = phy_informer.forward(
        {
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )
    phy_loss = physics_loss_dict["momentum_x"]

    print("Spectral: ", phy_loss.shape)
    plot_fields(phy_loss[0, 0], "spectral_momentum_x")

    # TEST Unstructured data, Least Squares Difference
    # Set the model meta data explicitly (for demo purposes only)
    model = UNet(output_types=["mesh_graph"], preferred_grad_method="least_squares")

    phy_informer = PhysicsInformer(
        required_outputs=["momentum_x"],
        grad_method="least_squares",
        equations=ns,
    )

    pred_outvar = model(coords_unstructured)
    physics_loss_dict = phy_informer.forward(
        {
            "coordinates": coords_unstructured,
            "nodes": node_ids,
            "edges": edges,
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )
    phy_loss = physics_loss_dict["momentum_x"]

    print("Mesh Graph: ", phy_loss.shape)
    plot_fields(phy_loss.reshape(steps, steps, steps), name="least_squares_momentum_x")

    # TEST Point Cloud data, Meshless Finite Difference
    # Set the model meta data explicitly (for demo purposes only)
    model = UNet(
        output_types=["point_cloud"], preferred_grad_method="meshless_finite_difference"
    )

    phy_informer = PhysicsInformer(
        required_outputs=["momentum_x"],
        equations=ns,
        grad_method="meshless_finite_difference",
        fd_dx=0.001,
    )

    pred_outvar = model(coords_unstructured)
    (
        po_posx,
        po_negx,
        po_posy,
        po_negy,
        po_posz,
        po_negz,
    ) = compute_stencil(coords_unstructured, model, dx=0.001)

    physics_loss_dict = phy_informer.forward(
        {
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
            "u>>x::1": po_posx[:, 0:1],
            "v>>x::1": po_posx[:, 1:2],
            "w>>x::1": po_posx[:, 2:3],
            "p>>x::1": po_posx[:, 3:4],
            "u>>x::-1": po_negx[:, 0:1],
            "v>>x::-1": po_negx[:, 1:2],
            "w>>x::-1": po_negx[:, 2:3],
            "p>>x::-1": po_negx[:, 3:4],
            "u>>y::1": po_posy[:, 0:1],
            "v>>y::1": po_posy[:, 1:2],
            "w>>y::1": po_posy[:, 2:3],
            "p>>y::1": po_posy[:, 3:4],
            "u>>y::-1": po_negy[:, 0:1],
            "v>>y::-1": po_negy[:, 1:2],
            "w>>y::-1": po_negy[:, 2:3],
            "p>>y::-1": po_negy[:, 3:4],
            "u>>z::1": po_posz[:, 0:1],
            "v>>z::1": po_posz[:, 1:2],
            "w>>z::1": po_posz[:, 2:3],
            "p>>z::1": po_posz[:, 3:4],
            "u>>z::-1": po_negz[:, 0:1],
            "v>>z::-1": po_negz[:, 1:2],
            "w>>z::-1": po_negz[:, 2:3],
            "p>>z::-1": po_negz[:, 3:4],
        },
    )
    phy_loss = physics_loss_dict["momentum_x"]

    print("MFD: ", phy_loss.shape)
    plot_fields(phy_loss.reshape(steps, steps, steps), name="mfd_momentum_x")

    # TEST Point Cloud data, AutoDiff
    # Set the model meta data explicitly (for demo purposes only)
    model = UNet(output_types=["point_cloud"], preferred_grad_method="autodiff")

    phy_informer = PhysicsInformer(
        required_outputs=["momentum_x"],
        grad_method="autodiff",
        equations=ns,
    )

    pred_outvar = model(coords_unstructured)
    physics_loss_dict = phy_informer.forward(
        {
            "coordinates": coords_unstructured,
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )
    phy_loss = physics_loss_dict["momentum_x"]

    print("Autodiff: ", phy_loss.shape)
    plot_fields(phy_loss.reshape(steps, steps, steps), name="autodiff_momentum_x")
