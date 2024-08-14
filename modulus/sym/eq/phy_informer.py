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
from modulus.sym.key import Key
from modulus.sym.graph import Graph

from typing import Dict, List, Set, Optional, Union, Callable
from modulus.sym.node import Node

from modulus.sym.eq.spatial_grads.spatial_grads import (
    GradientCalculator,
    compute_connectivity_tensor,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PhysicsInformer(object):
    def __init__(
        self,
        required_outputs,
        equations,
        grad_method,
        available_inputs=None,
        fd_dx=0.001,  # only applies for FD and Meshless FD. Ignored for the rest
        bounds=[2 * np.pi, 2 * np.pi, 2 * np.pi],
        device=None,
    ):
        super().__init__()
        self.available_inputs = available_inputs
        self.required_outputs = required_outputs
        self.equations = equations
        self.dim = equations.dim
        self.grad_method = grad_method
        self.fd_dx = fd_dx
        self.bounds = bounds
        self.device = device if device is not None else torch.device("cpu")
        self.grad_calc = GradientCalculator(device=self.device)
        self.nodes = self.equations.make_nodes()

        self.require_mixed_derivs = False

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

        return Graph(
            self.nodes, input_keys_sym, output_keys_sym, diff_nodes=diff_nodes
        ).to(self.device)

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
            self.require_mixed_derivs = True

        if str(derr).count("__") == 1:
            first_deriv.add(str(derr))
        elif str(derr).count("__") == 2:
            second_deriv.add(str(derr))

    def _create_diff_nodes(self, derivatives, dim, order):
        diff_nodes = []
        for derr_var in derivatives:
            node = self._create_diff_node(derr_var, dim, order)
            if node:
                diff_nodes.append(node)
        return diff_nodes

    def _create_diff_node(self, derr_var, dim, order):
        methods = {
            "finite_difference": self._fd_gradient_module,
            "spectral": self._spectral_gradient_module,
            "least_squares": self._ls_gradient_module,
            "autodiff": self._autodiff_gradient_module,
            "meshless_finite_difference": self._meshless_fd_gradient_module,
        }

        if self.grad_method in methods:
            return Node(
                [derr_var],
                self._derivative_keys(
                    derr_var, dim, order, return_mixed_derivs=self.require_mixed_derivs
                ),
                methods[self.grad_method](derr_var, dim, order),
            )

    def _derivative_keys(self, derr_var, dim, order, return_mixed_derivs=False):
        base_keys = ["__x", "__y", "__z"]
        base_keys = [base_keys[i] for i in range(dim)]
        output_keys = [f"{derr_var}{k * order}" for k in base_keys]
        if return_mixed_derivs:
            if order == 2:
                if dim == 2:
                    output_keys.append(f"{derr_var}__x__y")
                    output_keys.append(f"{derr_var}__y__x")
                if dim == 3:
                    output_keys.append(f"{derr_var}__x__y")
                    output_keys.append(f"{derr_var}__y__x")
                    output_keys.append(f"{derr_var}__x__z")
                    output_keys.append(f"{derr_var}__z__x")
                    output_keys.append(f"{derr_var}__y__z")
                    output_keys.append(f"{derr_var}__z__y")
        return output_keys

    def _fd_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            dx=self.fd_dx,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def _spectral_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            ell=self.bounds,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def _ls_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def _autodiff_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def _meshless_fd_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            dx=self.fd_dx,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def forward(self, inputs):
        if self.grad_method == "least_squares":
            connectivity_tensor = compute_connectivity_tensor(
                inputs["coordinates"], inputs["nodes"], inputs["edges"]
            )
            inputs["connectivity_tensor"] = connectivity_tensor

        return self.graph.forward(inputs)
