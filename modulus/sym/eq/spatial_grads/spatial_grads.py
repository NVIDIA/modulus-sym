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

import torch
import numpy as np
from typing import Dict, List, Set, Optional, Union, Callable
import logging
from modulus.sym.eq.mfd import grads as mfd_grads
from modulus.sym.eq.fd import grads as fd_grads
from modulus.sym.eq.derivatives import gradient_autodiff


def compute_stencil(coords, model, dx):
    # compute stencil points
    posx = coords[:, 0:1] + torch.ones_like(coords[:, 0:1]) * dx
    negx = coords[:, 0:1] - torch.ones_like(coords[:, 0:1]) * dx
    posy = coords[:, 1:2] + torch.ones_like(coords[:, 0:1]) * dx
    negy = coords[:, 1:2] - torch.ones_like(coords[:, 0:1]) * dx
    posz = coords[:, 2:3] + torch.ones_like(coords[:, 0:1]) * dx
    negz = coords[:, 2:3] - torch.ones_like(coords[:, 0:1]) * dx

    uposx = model(torch.cat([posx, coords[:, 1:2], coords[:, 2:3]], dim=1))
    unegx = model(torch.cat([negx, coords[:, 1:2], coords[:, 2:3]], dim=1))
    uposy = model(torch.cat([coords[:, 0:1], posy, coords[:, 2:3]], dim=1))
    unegy = model(torch.cat([coords[:, 0:1], negy, coords[:, 2:3]], dim=1))
    uposz = model(torch.cat([coords[:, 0:1], coords[:, 1:2], posz], dim=1))
    unegz = model(torch.cat([coords[:, 0:1], coords[:, 1:2], negz], dim=1))

    return uposx, unegx, uposy, unegy, uposz, unegz


def compute_connectivity_tensor(coords, nodes, edges):
    edge_list = []
    for i in range(edges.size(0)):
        node1, node2 = edges[i][0].item(), edges[i][1].item()
        edge_list.append(tuple(sorted((node1, node2))))
    unique_edges = set(edge_list)
    node_edges = {node.item(): [] for node in nodes}
    for edge in unique_edges:
        node1, node2 = edge
        if node1 in node_edges:
            node_edges[node1].append((node1, node2))
        if node2 in node_edges:
            node_edges[node2].append((node2, node1))
    avg_connectivity = 0.0
    max_connectivity = []
    for k, v in node_edges.items():
        avg_connectivity += len(v)
        max_connectivity.append(len(v))
    avg_connectivity = round(avg_connectivity / len(node_edges.keys()))
    max_connectivity = np.array(max_connectivity).max()
    for k, v in node_edges.items():
        if len(v) < max_connectivity:
            empty_list = [(0, 0) for _ in range(max_connectivity - len(v))]
            v = v + empty_list
            node_edges[k] = torch.tensor(v)
        elif len(v) > max_connectivity:
            v = v[0:max_connectivity]
            node_edges[k] = torch.tensor(v)
        else:
            node_edges[k] = torch.tensor(v)
    connectivity_tensor = (
        torch.stack([v for v in node_edges.values()], dim=0)
        .to(torch.long)
        .to(nodes.device)
    )

    return connectivity_tensor


class GradientsAutoDiff(torch.nn.Module):
    def __init__(self, invar: str, dim: int = 3, order: int = 1):
        super(GradientsAutoDiff, self).__init__()

        self.invar = invar
        self.dim = dim
        self.order = order

        assert self.order < 3, "Derivatives only upto 2nd order are supported"

    def forward(self, input_dict):
        y = input_dict[self.invar]
        x = input_dict["coordinates"]

        if not x.requires_grad:
            logging.warning(
                "The input tensor does not have requries_grad set to true, "
                + "the computations might be incorrect!"
            )

        assert (
            x.shape[1] == self.dim
        ), f"Expected shape (N, {self.dim}), but got {x.shape}"

        grad = gradient_autodiff(y, [x])

        result = {}
        axis_list = ["x", "y", "z"]
        if self.order == 1:
            for axis in range(self.dim):
                result[f"{self.invar}__{axis_list[axis]}"] = grad[0][:, axis : axis + 1]
        elif self.order == 2:
            ggrad = gradient_autodiff(grad[0], [x])
            for axis in range(self.dim):
                result[f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"] = ggrad[
                    0
                ][:, axis : axis + 1]
        return result


class GradientsMeshlessFiniteDifference(torch.nn.Module):
    def __init__(
        self, invar: str, dx: Union[Union[float, int]], dim: int = 3, order: int = 1
    ):
        super(GradientsMeshlessFiniteDifference, self).__init__()

        self.invar = invar
        self.dx = dx
        self.dim = dim
        self.order = order

        if isinstance(self.dx, (float, int)):
            self.dx = [self.dx for _ in range(self.dim)]

        assert self.order < 3, "Derivatives only upto 2nd order are supported"
        assert len(self.dx) == self.dim, f"Mismatch in {self.dim} and {self.dx}"

        self.init_derivative_operators()

    def init_derivative_operators(self):
        self.first_deriv_ops = {}
        self.second_deriv_ops = {}

        if self.order == 1:
            for axis in range(self.dim):
                axis_name = ["x", "y", "z"][axis]
                self.first_deriv_ops[axis] = mfd_grads.FirstDerivO2(
                    var=self.invar,
                    indep_var=axis_name,
                    out_name=f"{self.invar}__{axis_name}",
                )
        elif self.order == 2:
            for axis in range(self.dim):
                axis_name = ["x", "y", "z"][axis]
                self.second_deriv_ops[axis] = mfd_grads.SecondDerivO2(
                    var=self.invar,
                    indep_var=axis_name,
                    out_name=f"{self.invar}__{axis_name}__{axis_name}",
                )

    def forward(self, input_dict):
        result = {}
        axis_list = ["x", "y", "z"]
        if self.order == 1:
            for axis, op in self.first_deriv_ops.items():
                result[f"{self.invar}__{axis_list[axis]}"] = op.forward(
                    input_dict, self.dx[axis]
                )[f"{self.invar}__{axis_list[axis]}"]
        elif self.order == 2:
            for axis, op in self.second_deriv_ops.items():
                result[
                    f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"
                ] = op.forward(input_dict, self.dx[axis])[
                    f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"
                ]

        return result


class GradientsFiniteDifference(torch.nn.Module):
    def __init__(
        self,
        invar: str,
        dx: Union[Union[float, int], List[float]],
        dim: int = 3,
        order: int = 1,
    ):
        super(GradientsFiniteDifference, self).__init__()

        self.invar = invar
        self.dx = dx
        self.dim = dim
        self.order = order

        if isinstance(self.dx, (float, int)):
            self.dx = [self.dx for _ in range(self.dim)]

        assert self.order < 3, "Derivatives only upto 2nd order are supported"
        assert len(self.dx) == self.dim, f"Mismatch in {self.dim} and {self.dx}"

        if self.order == 1:
            self.deriv_modulue = fd_grads.FirstDerivO2(self.dim, self.dx)
        elif self.order == 2:
            self.deriv_modulue = fd_grads.SecondDerivO2(self.dim, self.dx)

    def forward(self, input_dict):
        u = input_dict[self.invar]

        assert (
            u.dim() - 2
        ) == self.dim, f"Expected a {self.dim + 2} dimensional tensor, but got {u.dim()} dimensional tensor"

        # compute finite difference based on convolutional operation
        result = {}
        derivatives = self.deriv_modulue(u)
        axis_list = ["x", "y", "z"]
        if self.order == 1:
            for axis, derivative in enumerate(derivatives):
                result[f"{self.invar}__{axis_list[axis]}"] = derivative
        elif self.order == 2:
            for axis, derivative in enumerate(derivatives):
                result[
                    f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"
                ] = derivative

        return result


class GradientsSpectral(torch.nn.Module):
    def __init__(
        self, invar: str, ell: Union[Union[float, int]], dim: int = 3, order: int = 1
    ):
        super(GradientsSpectral, self).__init__()

        self.invar = invar
        self.ell = ell
        self.dim = dim
        self.order = order

        if isinstance(self.ell, (float, int)):
            self.ell = [self.ell for _ in range(self.dim)]

        assert self.order < 3, "Derivatives only upto 2nd order are supported"
        assert len(self.ell) == self.dim, f"Mismatch in {self.dim} and {self.ell}"

    def forward(self, input_dict):
        u = input_dict[self.invar]

        pi = float(np.pi)

        n = tuple(u.shape[2:])
        assert (
            len(n) == self.dim
        ), f"Expected a {self.dim + 2} dimensional tensor, but got {u.dim()} dimensional tensor"

        # compute the fourier transform
        u_h = torch.fft.fftn(u, dim=list(range(2, self.dim + 2)))

        # make wavenumbers
        kx = []
        for i, nx in enumerate(n):
            kx.append(
                torch.cat(
                    (
                        torch.arange(start=0, end=nx // 2, step=1, device=u.device),
                        torch.arange(start=-nx // 2, end=0, step=1, device=u.device),
                    ),
                    0,
                ).reshape((i + 2) * [1] + [nx] + (self.dim - i - 1) * [1])
            )

        # compute laplacian in the fourier space
        j = torch.complex(
            torch.tensor([0.0], device=u.device), torch.tensor([1.0], device=u.device)
        )

        wx_h = [j * k_x_i * u_h * (2 * pi / self.ell[i]) for i, k_x_i in enumerate(kx)]

        result = {}
        axis_list = ["x", "y", "z"]
        if self.order == 1:
            # inverse fourier transform out
            wx = torch.cat(
                [
                    torch.fft.ifftn(wx_h_i, dim=list(range(2, self.dim + 2))).real
                    for wx_h_i in wx_h
                ],
                dim=1,
            )
            for axis in range(self.dim):
                result[f"{self.invar}__{axis_list[axis]}"] = wx[:, axis : axis + 1]

        elif self.order == 2:
            wxx_h = [
                j * k_x_i * wx_h_i * (2 * pi / self.ell[i])
                for i, (wx_h_i, k_x_i) in enumerate(zip(wx_h, kx))
            ]

            # inverse fourier transform out
            wxx = torch.cat(
                [
                    torch.fft.ifftn(wxx_h_i, dim=list(range(2, self.dim + 2))).real
                    for wxx_h_i in wxx_h
                ],
                dim=1,
            )

            for axis in range(self.dim):
                result[f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"] = wxx[
                    :, axis : axis + 1
                ]

        return result


class GradientsLeastSquares(torch.nn.Module):
    def __init__(self, invar: str, dim: int = 3, order: int = 1):
        super(GradientsLeastSquares, self).__init__()
        self.invar = invar
        self.cache = {}
        self.dim = dim
        self.order = order

        assert (
            self.dim > 1
        ), "1D gradients using Least squares is not supported. Please try other methods."
        assert self.order < 3, "Derivatives only upto 2nd order are supported"

    def forward(self, input_dict):

        coords = input_dict["coordinates"]

        assert (
            coords.shape[1] == self.dim
        ), f"Expected shape (N, {self.dim}), but got {coords.shape}"

        connectivity_tensor = input_dict["connectivity_tensor"]
        p1 = coords[connectivity_tensor[:, :, 0]]
        p2 = coords[connectivity_tensor[:, :, 1]]

        if self.dim == 2:
            dx = p1[:, :, 0] - p2[:, :, 0]
            dy = p1[:, :, 1] - p2[:, :, 1]
            y = input_dict[self.invar]

            f1 = y[connectivity_tensor[:, :, 0]]
            f2 = y[connectivity_tensor[:, :, 1]]

            du = (f1 - f2).squeeze(-1)
            w = 1 / torch.sqrt(dx**2 + dy**2)
            w = torch.where(torch.isinf(w), torch.tensor(1.0).to(w.device), w)
            # mask = ~((dx == 0) & (dy == 0) & (dz == 0))
            mask = torch.ones_like(dx)

            if self.order == 1:
                # compute the gradients using either cramers rule or qr decomposion
                a1 = torch.sum((w**2 * dx * dx) * mask, dim=1)
                b1 = torch.sum((w**2 * dx * dy) * mask, dim=1)
                d1 = torch.sum((w**2 * du * dx) * mask, dim=1)

                a2 = torch.sum((w**2 * dx * dy) * mask, dim=1)
                b2 = torch.sum((w**2 * dy * dy) * mask, dim=1)
                d2 = torch.sum((w**2 * du * dy) * mask, dim=1)

                detA = torch.linalg.det(
                    torch.stack(
                        [
                            torch.stack([a1, a2], dim=1),
                            torch.stack([b1, b2], dim=1),
                        ],
                        dim=2,
                    )
                )
                dudx = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack([d1, d2], dim=1),
                                torch.stack([b1, b2], dim=1),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )
                dudy = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack([a1, a2], dim=1),
                                torch.stack([d1, d2], dim=1),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )
                return {
                    f"{self.invar}__x": dudx.unsqueeze(dim=1),
                    f"{self.invar}__y": dudy.unsqueeze(dim=1),
                }
            elif self.order == 2:
                a1 = torch.sum((w**2 * dx * dx) * mask, dim=1)
                b1 = torch.sum((w**2 * dx * dy) * mask, dim=1)
                d1 = torch.sum((0.5 * w**2 * dx * dx * dx) * mask, dim=1)
                e1 = torch.sum((0.5 * w**2 * dx * dy * dy) * mask, dim=1)
                g1 = torch.sum((w**2 * dx * dx * dy) * mask, dim=1)
                j1 = torch.sum((w**2 * du * dx) * mask, dim=1)

                a2 = torch.sum((w**2 * dx * dy) * mask, dim=1)
                b2 = torch.sum((w**2 * dy * dy) * mask, dim=1)
                d2 = torch.sum((0.5 * w**2 * dx * dx * dy) * mask, dim=1)
                e2 = torch.sum((0.5 * w**2 * dy * dy * dy) * mask, dim=1)
                g2 = torch.sum((w**2 * dx * dy * dy) * mask, dim=1)
                j2 = torch.sum((w**2 * du * dy) * mask, dim=1)

                a4 = torch.sum((0.5 * w**2 * dx * dx * dx) * mask, dim=1)
                b4 = torch.sum((0.5 * w**2 * dx * dx * dy) * mask, dim=1)
                d4 = torch.sum((0.25 * w**2 * dx * dx * dx * dx) * mask, dim=1)
                e4 = torch.sum((0.25 * w**2 * dx * dx * dy * dy) * mask, dim=1)
                g4 = torch.sum((0.5 * w**2 * dx * dx * dx * dy) * mask, dim=1)
                j4 = torch.sum((0.5 * w**2 * du * dx * dx) * mask, dim=1)

                a5 = torch.sum((0.5 * w**2 * dx * dy * dy) * mask, dim=1)
                b5 = torch.sum((0.5 * w**2 * dy * dy * dy) * mask, dim=1)
                d5 = torch.sum((0.25 * w**2 * dx * dx * dy * dy) * mask, dim=1)
                e5 = torch.sum((0.25 * w**2 * dy * dy * dy * dy) * mask, dim=1)
                g5 = torch.sum((0.5 * w**2 * dx * dy * dy * dy) * mask, dim=1)
                j5 = torch.sum((0.5 * w**2 * du * dy * dy) * mask, dim=1)

                a7 = torch.sum((w**2 * dx * dx * dy) * mask, dim=1)
                b7 = torch.sum((w**2 * dx * dy * dy) * mask, dim=1)
                d7 = torch.sum((0.5 * w**2 * dx * dx * dx * dy) * mask, dim=1)
                e7 = torch.sum((0.5 * w**2 * dx * dy * dy * dy) * mask, dim=1)
                g7 = torch.sum((w**2 * dx * dx * dy * dy) * mask, dim=1)
                j7 = torch.sum((w**2 * du * dx * dy) * mask, dim=1)

                matA = torch.stack(
                    [
                        torch.stack([a1, a2, a4, a5, a7], dim=1),
                        torch.stack([b1, b2, b4, b5, b7], dim=1),
                        torch.stack([d1, d2, d4, d5, d7], dim=1),
                        torch.stack([e1, e2, e4, e5, e7], dim=1),
                        torch.stack([g1, g2, g4, g5, g7], dim=1),
                    ],
                    dim=2,
                )

                detA = torch.linalg.det(matA)
                detA = detA + 1e-10

                d2udx2 = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack([a1, a2, a4, a5, a7], dim=1),
                                torch.stack([b1, b2, b4, b5, b7], dim=1),
                                torch.stack([j1, j2, j4, j5, j7], dim=1),
                                torch.stack([e1, e2, e4, e5, e7], dim=1),
                                torch.stack([g1, g2, g4, g5, g7], dim=1),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )

                d2udy2 = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack([a1, a2, a4, a5, a7], dim=1),
                                torch.stack([b1, b2, b4, b5, b7], dim=1),
                                torch.stack([d1, d2, d4, d5, d7], dim=1),
                                torch.stack([j1, j2, j4, j5, j7], dim=1),
                                torch.stack([g1, g2, g4, g5, g7], dim=1),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )

                return {
                    f"{self.invar}__x__x": d2udx2.unsqueeze(dim=1),
                    f"{self.invar}__y__y": d2udy2.unsqueeze(dim=1),
                }

        elif self.dim == 3:
            dx = p1[:, :, 0] - p2[:, :, 0]
            dy = p1[:, :, 1] - p2[:, :, 1]
            dz = p1[:, :, 2] - p2[:, :, 2]
            y = input_dict[self.invar]

            f1 = y[connectivity_tensor[:, :, 0]]
            f2 = y[connectivity_tensor[:, :, 1]]

            du = (f1 - f2).squeeze(-1)
            w = 1 / torch.sqrt(dx**2 + dy**2 + dz**2)
            w = torch.where(torch.isinf(w), torch.tensor(1.0).to(w.device), w)
            # mask = ~((dx == 0) & (dy == 0) & (dz == 0))
            mask = torch.ones_like(dx)

            if self.order == 1:
                # compute the gradients using either cramers rule or qr decomposion
                a1 = torch.sum((w**2 * dx * dx) * mask, dim=1)
                b1 = torch.sum((w**2 * dx * dy) * mask, dim=1)
                c1 = torch.sum((w**2 * dx * dz) * mask, dim=1)
                d1 = torch.sum((w**2 * du * dx) * mask, dim=1)

                a2 = torch.sum((w**2 * dx * dy) * mask, dim=1)
                b2 = torch.sum((w**2 * dy * dy) * mask, dim=1)
                c2 = torch.sum((w**2 * dy * dz) * mask, dim=1)
                d2 = torch.sum((w**2 * du * dy) * mask, dim=1)

                a3 = torch.sum((w**2 * dx * dz) * mask, dim=1)
                b3 = torch.sum((w**2 * dy * dz) * mask, dim=1)
                c3 = torch.sum((w**2 * dz * dz) * mask, dim=1)
                d3 = torch.sum((w**2 * du * dz) * mask, dim=1)

                detA = torch.linalg.det(
                    torch.stack(
                        [
                            torch.stack([a1, a2, a3], dim=1),
                            torch.stack([b1, b2, b3], dim=1),
                            torch.stack([c1, c2, c3], dim=1),
                        ],
                        dim=2,
                    )
                )
                dudx = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack([d1, d2, d3], dim=1),
                                torch.stack([b1, b2, b3], dim=1),
                                torch.stack([c1, c2, c3], dim=1),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )
                dudy = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack([a1, a2, a3], dim=1),
                                torch.stack([d1, d2, d3], dim=1),
                                torch.stack([c1, c2, c3], dim=1),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )
                dudz = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack([a1, a2, a3], dim=1),
                                torch.stack([b1, b2, b3], dim=1),
                                torch.stack([d1, d2, d3], dim=1),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )

                return {
                    f"{self.invar}__x": dudx.unsqueeze(dim=1),
                    f"{self.invar}__y": dudy.unsqueeze(dim=1),
                    f"{self.invar}__z": dudz.unsqueeze(dim=1),
                }
            elif self.order == 2:
                a1 = torch.sum((w**2 * dx * dx) * mask, dim=1)
                b1 = torch.sum((w**2 * dx * dy) * mask, dim=1)
                c1 = torch.sum((w**2 * dx * dz) * mask, dim=1)
                d1 = torch.sum((0.5 * w**2 * dx * dx * dx) * mask, dim=1)
                e1 = torch.sum((0.5 * w**2 * dx * dy * dy) * mask, dim=1)
                f1 = torch.sum((0.5 * w**2 * dx * dz * dz) * mask, dim=1)
                g1 = torch.sum((w**2 * dx * dx * dy) * mask, dim=1)
                h1 = torch.sum((w**2 * dx * dy * dz) * mask, dim=1)
                i1 = torch.sum((w**2 * dx * dx * dz) * mask, dim=1)
                j1 = torch.sum((w**2 * du * dx) * mask, dim=1)

                a2 = torch.sum((w**2 * dx * dy) * mask, dim=1)
                b2 = torch.sum((w**2 * dy * dy) * mask, dim=1)
                c2 = torch.sum((w**2 * dy * dz) * mask, dim=1)
                d2 = torch.sum((0.5 * w**2 * dx * dx * dy) * mask, dim=1)
                e2 = torch.sum((0.5 * w**2 * dy * dy * dy) * mask, dim=1)
                f2 = torch.sum((0.5 * w**2 * dy * dz * dz) * mask, dim=1)
                g2 = torch.sum((w**2 * dx * dy * dy) * mask, dim=1)
                h2 = torch.sum((w**2 * dy * dy * dz) * mask, dim=1)
                i2 = torch.sum((w**2 * dx * dy * dz) * mask, dim=1)
                j2 = torch.sum((w**2 * du * dy) * mask, dim=1)

                a3 = torch.sum((w**2 * dx * dz) * mask, dim=1)
                b3 = torch.sum((w**2 * dy * dz) * mask, dim=1)
                c3 = torch.sum((w**2 * dz * dz) * mask, dim=1)
                d3 = torch.sum((0.5 * w**2 * dx * dx * dz) * mask, dim=1)
                e3 = torch.sum((0.5 * w**2 * dy * dy * dz) * mask, dim=1)
                f3 = torch.sum((0.5 * w**2 * dz * dz * dz) * mask, dim=1)
                g3 = torch.sum((w**2 * dx * dy * dz) * mask, dim=1)
                h3 = torch.sum((w**2 * dy * dz * dz) * mask, dim=1)
                i3 = torch.sum((w**2 * dx * dz * dz) * mask, dim=1)
                j3 = torch.sum((w**2 * du * dz) * mask, dim=1)

                a4 = torch.sum((0.5 * w**2 * dx * dx * dx) * mask, dim=1)
                b4 = torch.sum((0.5 * w**2 * dx * dx * dy) * mask, dim=1)
                c4 = torch.sum((0.5 * w**2 * dx * dx * dz) * mask, dim=1)
                d4 = torch.sum((0.25 * w**2 * dx * dx * dx * dx) * mask, dim=1)
                e4 = torch.sum((0.25 * w**2 * dx * dx * dy * dy) * mask, dim=1)
                f4 = torch.sum((0.25 * w**2 * dx * dx * dz * dz) * mask, dim=1)
                g4 = torch.sum((0.5 * w**2 * dx * dx * dx * dy) * mask, dim=1)
                h4 = torch.sum((0.5 * w**2 * dx * dx * dy * dz) * mask, dim=1)
                i4 = torch.sum((0.5 * w**2 * dx * dx * dx * dz) * mask, dim=1)
                j4 = torch.sum((0.5 * w**2 * du * dx * dx) * mask, dim=1)

                a5 = torch.sum((0.5 * w**2 * dx * dy * dy) * mask, dim=1)
                b5 = torch.sum((0.5 * w**2 * dy * dy * dy) * mask, dim=1)
                c5 = torch.sum((0.5 * w**2 * dy * dy * dz) * mask, dim=1)
                d5 = torch.sum((0.25 * w**2 * dx * dx * dy * dy) * mask, dim=1)
                e5 = torch.sum((0.25 * w**2 * dy * dy * dy * dy) * mask, dim=1)
                f5 = torch.sum((0.25 * w**2 * dy * dy * dz * dz) * mask, dim=1)
                g5 = torch.sum((0.5 * w**2 * dx * dy * dy * dy) * mask, dim=1)
                h5 = torch.sum((0.5 * w**2 * dy * dy * dy * dz) * mask, dim=1)
                i5 = torch.sum((0.5 * w**2 * dx * dy * dy * dz) * mask, dim=1)
                j5 = torch.sum((0.5 * w**2 * du * dy * dy) * mask, dim=1)

                a6 = torch.sum((0.5 * w**2 * dx * dz * dz) * mask, dim=1)
                b6 = torch.sum((0.5 * w**2 * dy * dz * dz) * mask, dim=1)
                c6 = torch.sum((0.5 * w**2 * dz * dz * dz) * mask, dim=1)
                d6 = torch.sum((0.25 * w**2 * dx * dx * dz * dz) * mask, dim=1)
                e6 = torch.sum((0.25 * w**2 * dy * dy * dz * dz) * mask, dim=1)
                f6 = torch.sum((0.25 * w**2 * dz * dz * dz * dz) * mask, dim=1)
                g6 = torch.sum((0.5 * w**2 * dx * dy * dz * dz) * mask, dim=1)
                h6 = torch.sum((0.5 * w**2 * dy * dz * dz * dz) * mask, dim=1)
                i6 = torch.sum((0.5 * w**2 * dx * dz * dz * dz) * mask, dim=1)
                j6 = torch.sum((0.5 * w**2 * du * dz * dz) * mask, dim=1)

                a7 = torch.sum((w**2 * dx * dx * dy) * mask, dim=1)
                b7 = torch.sum((w**2 * dx * dy * dy) * mask, dim=1)
                c7 = torch.sum((w**2 * dx * dy * dz) * mask, dim=1)
                d7 = torch.sum((0.5 * w**2 * dx * dx * dx * dy) * mask, dim=1)
                e7 = torch.sum((0.5 * w**2 * dx * dy * dy * dy) * mask, dim=1)
                f7 = torch.sum((0.5 * w**2 * dx * dy * dz * dz) * mask, dim=1)
                g7 = torch.sum((w**2 * dx * dx * dy * dy) * mask, dim=1)
                h7 = torch.sum((w**2 * dx * dy * dy * dz) * mask, dim=1)
                i7 = torch.sum((w**2 * dx * dx * dy * dz) * mask, dim=1)
                j7 = torch.sum((w**2 * du * dx * dy) * mask, dim=1)

                a8 = torch.sum((w**2 * dx * dy * dz) * mask, dim=1)
                b8 = torch.sum((w**2 * dy * dy * dz) * mask, dim=1)
                c8 = torch.sum((w**2 * dy * dz * dz) * mask, dim=1)
                d8 = torch.sum((0.5 * w**2 * dx * dx * dy * dz) * mask, dim=1)
                e8 = torch.sum((0.5 * w**2 * dy * dy * dy * dz) * mask, dim=1)
                f8 = torch.sum((0.5 * w**2 * dy * dz * dz * dz) * mask, dim=1)
                g8 = torch.sum((w**2 * dx * dy * dy * dz) * mask, dim=1)
                h8 = torch.sum((w**2 * dy * dy * dz * dz) * mask, dim=1)
                i8 = torch.sum((w**2 * dx * dy * dz * dz) * mask, dim=1)
                j8 = torch.sum((w**2 * du * dy * dz) * mask, dim=1)

                a9 = torch.sum((w**2 * dx * dx * dz) * mask, dim=1)
                b9 = torch.sum((w**2 * dx * dy * dz) * mask, dim=1)
                c9 = torch.sum((w**2 * dx * dz * dz) * mask, dim=1)
                d9 = torch.sum((0.5 * w**2 * dx * dx * dx * dz) * mask, dim=1)
                e9 = torch.sum((0.5 * w**2 * dx * dy * dy * dz) * mask, dim=1)
                f9 = torch.sum((0.5 * w**2 * dx * dz * dz * dz) * mask, dim=1)
                g9 = torch.sum((w**2 * dx * dx * dy * dz) * mask, dim=1)
                h9 = torch.sum((w**2 * dx * dy * dz * dz) * mask, dim=1)
                i9 = torch.sum((w**2 * dx * dx * dz * dz) * mask, dim=1)
                j9 = torch.sum((w**2 * du * dx * dz) * mask, dim=1)

                matA = torch.stack(
                    [
                        torch.stack([a1, a2, a3, a4, a5, a6, a7, a8, a9], dim=1),
                        torch.stack([b1, b2, b3, b4, b5, b6, b7, b8, b9], dim=1),
                        torch.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1),
                        torch.stack([d1, d2, d3, d4, d5, d6, d7, d8, d9], dim=1),
                        torch.stack([e1, e2, e3, e4, e5, e6, e7, e8, e9], dim=1),
                        torch.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9], dim=1),
                        torch.stack([g1, g2, g3, g4, g5, g6, g7, g8, g9], dim=1),
                        torch.stack([h1, h2, h3, h4, h5, h6, h7, h8, h9], dim=1),
                        torch.stack([i1, i2, i3, i4, i5, i6, i7, i8, i9], dim=1),
                    ],
                    dim=2,
                )

                detA = torch.linalg.det(matA)
                detA = detA + 1e-10

                d2udx2 = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack(
                                    [a1, a2, a3, a4, a5, a6, a7, a8, a9], dim=1
                                ),
                                torch.stack(
                                    [b1, b2, b3, b4, b5, b6, b7, b8, b9], dim=1
                                ),
                                torch.stack(
                                    [c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1
                                ),
                                torch.stack(
                                    [j1, j2, j3, j4, j5, j6, j7, j8, j9], dim=1
                                ),
                                torch.stack(
                                    [e1, e2, e3, e4, e5, e6, e7, e8, e9], dim=1
                                ),
                                torch.stack(
                                    [f1, f2, f3, f4, f5, f6, f7, f8, f9], dim=1
                                ),
                                torch.stack(
                                    [g1, g2, g3, g4, g5, g6, g7, g8, g9], dim=1
                                ),
                                torch.stack(
                                    [h1, h2, h3, h4, h5, h6, h7, h8, h9], dim=1
                                ),
                                torch.stack(
                                    [i1, i2, i3, i4, i5, i6, i7, i8, i9], dim=1
                                ),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )

                d2udy2 = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack(
                                    [a1, a2, a3, a4, a5, a6, a7, a8, a9], dim=1
                                ),
                                torch.stack(
                                    [b1, b2, b3, b4, b5, b6, b7, b8, b9], dim=1
                                ),
                                torch.stack(
                                    [c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1
                                ),
                                torch.stack(
                                    [d1, d2, d3, d4, d5, d6, d7, d8, d9], dim=1
                                ),
                                torch.stack(
                                    [j1, j2, j3, j4, j5, j6, j7, j8, j9], dim=1
                                ),
                                torch.stack(
                                    [f1, f2, f3, f4, f5, f6, f7, f8, f9], dim=1
                                ),
                                torch.stack(
                                    [g1, g2, g3, g4, g5, g6, g7, g8, g9], dim=1
                                ),
                                torch.stack(
                                    [h1, h2, h3, h4, h5, h6, h7, h8, h9], dim=1
                                ),
                                torch.stack(
                                    [i1, i2, i3, i4, i5, i6, i7, i8, i9], dim=1
                                ),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )

                d2udz2 = (
                    torch.linalg.det(
                        torch.stack(
                            [
                                torch.stack(
                                    [a1, a2, a3, a4, a5, a6, a7, a8, a9], dim=1
                                ),
                                torch.stack(
                                    [b1, b2, b3, b4, b5, b6, b7, b8, b9], dim=1
                                ),
                                torch.stack(
                                    [c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1
                                ),
                                torch.stack(
                                    [d1, d2, d3, d4, d5, d6, d7, d8, d9], dim=1
                                ),
                                torch.stack(
                                    [e1, e2, e3, e4, e5, e6, e7, e8, e9], dim=1
                                ),
                                torch.stack(
                                    [j1, j2, j3, j4, j5, j6, j7, j8, j9], dim=1
                                ),
                                torch.stack(
                                    [g1, g2, g3, g4, g5, g6, g7, g8, g9], dim=1
                                ),
                                torch.stack(
                                    [h1, h2, h3, h4, h5, h6, h7, h8, h9], dim=1
                                ),
                                torch.stack(
                                    [i1, i2, i3, i4, i5, i6, i7, i8, i9], dim=1
                                ),
                            ],
                            dim=2,
                        )
                    )
                    / detA
                )

                return {
                    f"{self.invar}__x__x": d2udx2.unsqueeze(dim=1),
                    f"{self.invar}__y__y": d2udy2.unsqueeze(dim=1),
                    f"{self.invar}__z__z": d2udz2.unsqueeze(dim=1),
                }


class GradientCalculator:
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cpu")
        self.methods = {}
        self._register_methods()

    def _register_methods(self):
        self.methods["autodiff"] = GradientsAutoDiff
        self.methods["meshless_finite_difference"] = GradientsMeshlessFiniteDifference
        self.methods["finite_difference"] = GradientsFiniteDifference
        self.methods["spectral"] = GradientsSpectral
        self.methods["least_squares"] = GradientsLeastSquares

    def get_gradient_module(self, method_name, invar, **kwargs):
        module = self.methods[method_name](invar, **kwargs)
        module.to(self.device)
        return module

    def compute_gradients(self, input_dict, method_name=None, invar=None, **kwargs):
        module = self.get_gradient_module(method_name, invar, **kwargs)
        return module.forward(input_dict)

    @staticmethod
    def clip_gradients():
        pass

    @staticmethod
    def visualize():
        pass

    @staticmethod
    def scale_grads():
        pass

    @staticmethod
    def unscale_grads():
        pass
