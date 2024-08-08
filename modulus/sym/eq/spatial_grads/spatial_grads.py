import torch
import numpy as np
from typing import Dict, List, Set, Optional, Union, Callable
import logging
from modulus.sym.eq.mfd.grads import FirstDerivO2, SecondDerivO2
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

        if self.order == 1:
            if self.dim == 1:
                return {
                    f"{self.invar}__x": grad[0][:, 0:1],
                }
            elif self.dim == 2:
                return {
                    f"{self.invar}__x": grad[0][:, 0:1],
                    f"{self.invar}__y": grad[0][:, 1:2],
                }
            elif self.dim == 3:
                return {
                    f"{self.invar}__x": grad[0][:, 0:1],
                    f"{self.invar}__y": grad[0][:, 1:2],
                    f"{self.invar}__z": grad[0][:, 2:3],
                }

        elif self.order == 2:
            ggrad = gradient_autodiff(grad[0], [x])

            if self.dim == 1:
                return {
                    f"{self.invar}__x__x": ggrad[0][:, 0:1],
                }
            elif self.dim == 2:
                return {
                    f"{self.invar}__x__x": ggrad[0][:, 0:1],
                    f"{self.invar}__y__y": ggrad[0][:, 1:2],
                }
            elif self.dim == 3:
                return {
                    f"{self.invar}__x__x": ggrad[0][:, 0:1],
                    f"{self.invar}__y__y": ggrad[0][:, 1:2],
                    f"{self.invar}__z__z": ggrad[0][:, 2:3],
                }


class GradientsMeshlessFiniteDifference(torch.nn.Module):
    def __init__(
        self, invar: str, dx: Union[Union[float, int]], dim: int = 3, order: int = 1
    ):
        super(GradientsMeshlessFiniteDifference, self).__init__()

        self.invar = invar
        self.dx = dx
        self.dim = dim
        self.order = order

        if self.dim == 1:
            if self.order == 1:
                self.first_deriv_x = FirstDerivO2(var=self.invar, indep_var="x", out_name=f"{self.invar}__x")
            if self.order == 2:
                self.second_deriv_x = SecondDerivO2(var=self.invar, indep_var="x", out_name=f"{self.invar}__x__x")
        if self.dim == 2:
            if self.order == 1:
                self.first_deriv_x = FirstDerivO2(var=self.invar, indep_var="x", out_name=f"{self.invar}__x")
                self.first_deriv_y = FirstDerivO2(var=self.invar, indep_var="y", out_name=f"{self.invar}__y")
            if self.order == 2:
                self.second_deriv_x = SecondDerivO2(var=self.invar, indep_var="x", out_name=f"{self.invar}__x__x")
                self.second_deriv_y = SecondDerivO2(var=self.invar, indep_var="y", out_name=f"{self.invar}__y__y")
        if self.dim == 3:
            if self.order == 1:
                self.first_deriv_x = FirstDerivO2(var=self.invar, indep_var="x", out_name=f"{self.invar}__x")
                self.first_deriv_y = FirstDerivO2(var=self.invar, indep_var="y", out_name=f"{self.invar}__y")
                self.first_deriv_z = FirstDerivO2(var=self.invar, indep_var="z", out_name=f"{self.invar}__z")
            if self.order == 2:
                self.second_deriv_x = SecondDerivO2(var=self.invar, indep_var="x", out_name=f"{self.invar}__x__x")
                self.second_deriv_y = SecondDerivO2(var=self.invar, indep_var="y", out_name=f"{self.invar}__y__y")
                self.second_deriv_z = SecondDerivO2(var=self.invar, indep_var="z", out_name=f"{self.invar}__z__z")
                
        if isinstance(self.dx, (float, int)):
            self.dx = [self.dx for _ in range(self.dim)]

        assert self.order < 3, "Derivatives only upto 2nd order are supported"
        assert len(self.dx) == self.dim, f"Mismatch in {self.dim} and {self.dx}"

    def forward(self, input_dict):
        if self.order == 1:
            # combine them according to the FD rules
            if self.dim == 1:
                return {
                    f"{self.invar}__x": self.first_deriv_x.forward(input_dict, self.dx[0])[f"{self.invar}__x"],
                }
            elif self.dim == 2:
                return {
                    f"{self.invar}__x": self.first_deriv_x.forward(input_dict, self.dx[0])[f"{self.invar}__x"],
                    f"{self.invar}__y": self.first_deriv_y.forward(input_dict, self.dx[1])[f"{self.invar}__y"],
                }
            elif self.dim == 3:
                return {
                    f"{self.invar}__x": self.first_deriv_x.forward(input_dict, self.dx[0])[f"{self.invar}__x"],
                    f"{self.invar}__y": self.first_deriv_y.forward(input_dict, self.dx[1])[f"{self.invar}__y"],
                    f"{self.invar}__z": self.first_deriv_z.forward(input_dict, self.dx[0])[f"{self.invar}__z"],
                }

        elif self.order == 2:
            # combine them according to the FD rules
            if self.dim == 1:
                return {
                    f"{self.invar}__x__x": self.second_deriv_x.forward(input_dict, self.dx[0])[f"{self.invar}__x__x"],
                }
            elif self.dim == 2:
                return {
                    f"{self.invar}__x__x": self.second_deriv_x.forward(input_dict, self.dx[0])[f"{self.invar}__x__x"],
                    f"{self.invar}__y__y": self.second_deriv_y.forward(input_dict, self.dx[1])[f"{self.invar}__y__y"],
                }
            elif self.dim == 3:
                return {
                    f"{self.invar}__x__x": self.second_deriv_x.forward(input_dict, self.dx[0])[f"{self.invar}__x__x"],
                    f"{self.invar}__y__y": self.second_deriv_y.forward(input_dict, self.dx[1])[f"{self.invar}__y__y"],
                    f"{self.invar}__z__z": self.second_deriv_z.forward(input_dict, self.dx[2])[f"{self.invar}__z__z"],
                }


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
        self.register_buffer(
            "ddx1D",
            torch.Tensor(
                [
                    -1.0 / 2.0,
                    0.0,
                    1.0 / 2.0,
                ]
            ),
        )
        self.register_buffer(
            "d2dx21D",
            torch.Tensor(
                [
                    1.0,
                    -2.0,
                    1.0,
                ]
            ),
        )

    def forward(self, input_dict):
        u = input_dict[self.invar]

        assert (
            u.dim() - 2
        ) == self.dim, f"Expected a {self.dim + 2} dimensional tensor, but got {u.dim()} dimensional tensor"

        u = torch.nn.functional.pad(u, self.dim * (1, 1), "replicate")

        # compute finite difference based on convolutional operation
        if self.order == 1:
            if self.dim == 1:
                ddx2D = torch.reshape(
                    self.ddx1D, shape=[1, 1] + 0 * [1] + [-1] + (0 - 0) * [1]
                )

                # return the grads
                dudx = torch.nn.functional.conv3d(
                    u, ddx2D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    :,
                ]
                dudx = (1 / self.dx[0]) * dudx

                return {
                    f"{self.invar}__x": dudx,
                }
            elif self.dim == 2:
                ddx2D = torch.reshape(
                    self.ddx1D, shape=[1, 1] + 0 * [1] + [-1] + (1 - 0) * [1]
                )
                ddy2D = torch.reshape(
                    self.ddx1D, shape=[1, 1] + 1 * [1] + [-1] + (1 - 1) * [1]
                )

                # return the grads
                dudx = torch.nn.functional.conv3d(
                    u, ddx2D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    :,
                    (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                ]
                dudx = (1 / self.dx[0]) * dudx

                dudy = torch.nn.functional.conv3d(
                    u, ddy2D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                    :,
                ]
                dudy = (1 / self.dx[1]) * dudy

                return {
                    f"{self.invar}__x": dudx,
                    f"{self.invar}__y": dudy,
                }

            elif self.dim == 3:
                ddx3D = torch.reshape(
                    self.ddx1D, shape=[1, 1] + 0 * [1] + [-1] + (2 - 0) * [1]
                )
                ddy3D = torch.reshape(
                    self.ddx1D, shape=[1, 1] + 1 * [1] + [-1] + (2 - 1) * [1]
                )
                ddz3D = torch.reshape(
                    self.ddx1D, shape=[1, 1] + 2 * [1] + [-1] + (2 - 2) * [1]
                )

                # return the grads
                dudx = torch.nn.functional.conv3d(
                    u, ddx3D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    :,
                    (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                    (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                ]
                dudx = (1 / self.dx[0]) * dudx

                dudy = torch.nn.functional.conv3d(
                    u, ddy3D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                    :,
                    (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                ]
                dudy = (1 / self.dx[1]) * dudy

                dudz = torch.nn.functional.conv3d(
                    u, ddz3D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                    (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                    :,
                ]
                dudz = (1 / self.dx[2]) * dudz

                return {
                    f"{self.invar}__x": dudx,
                    f"{self.invar}__y": dudy,
                    f"{self.invar}__z": dudz,
                }
        elif self.order == 2:
            if self.dim == 1:
                d2dx22D = torch.reshape(
                    self.d2dx21D, shape=[1, 1] + 0 * [1] + [-1] + (0 - 0) * [1]
                )

                # return the grads
                d2udx2 = torch.nn.functional.conv3d(
                    u, d2dx22D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    :,
                ]
                d2udx2 = (1 / (self.dx[0] ** 2)) * d2udx2

                return {
                    f"{self.invar}__x__x": d2udx2,
                }
            elif self.dim == 2:
                d2dx22D = torch.reshape(
                    self.d2dx21D, shape=[1, 1] + 0 * [1] + [-1] + (1 - 0) * [1]
                )
                d2dy22D = torch.reshape(
                    self.d2dx21D, shape=[1, 1] + 1 * [1] + [-1] + (1 - 1) * [1]
                )

                # return the grads
                d2udx2 = torch.nn.functional.conv3d(
                    u, d2dx22D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    :,
                    (self.d2dx21D.shape[0] - 1)
                    // 2 : -(self.d2dx21D.shape[0] - 1)
                    // 2,
                ]
                d2udx2 = (1 / (self.dx[0] ** 2)) * d2udx2

                d2udy2 = torch.nn.functional.conv3d(
                    u, d2dy22D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    (self.d2dx21D.shape[0] - 1)
                    // 2 : -(self.d2dx21D.shape[0] - 1)
                    // 2,
                    :,
                ]
                d2udy2 = (1 / (self.dx[1] ** 2)) * d2udy2

                return {
                    f"{self.invar}__x__x": d2udx2,
                    f"{self.invar}__y__y": d2udy2,
                }

            elif self.dim == 3:
                d2dx23D = torch.reshape(
                    self.d2dx21D, shape=[1, 1] + 0 * [1] + [-1] + (2 - 0) * [1]
                )
                d2dy23D = torch.reshape(
                    self.d2dx21D, shape=[1, 1] + 1 * [1] + [-1] + (2 - 1) * [1]
                )
                d2dz23D = torch.reshape(
                    self.d2dx21D, shape=[1, 1] + 2 * [1] + [-1] + (2 - 2) * [1]
                )

                # return the grads
                d2udx2 = torch.nn.functional.conv3d(
                    u, d2dx23D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    :,
                    (self.d2dx21D.shape[0] - 1)
                    // 2 : -(self.d2dx21D.shape[0] - 1)
                    // 2,
                    (self.d2dx21D.shape[0] - 1)
                    // 2 : -(self.d2dx21D.shape[0] - 1)
                    // 2,
                ]
                d2udx2 = (1 / (self.dx[0] ** 2)) * d2udx2

                d2udy2 = torch.nn.functional.conv3d(
                    u, d2dy23D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    (self.d2dx21D.shape[0] - 1)
                    // 2 : -(self.d2dx21D.shape[0] - 1)
                    // 2,
                    :,
                    (self.d2dx21D.shape[0] - 1)
                    // 2 : -(self.d2dx21D.shape[0] - 1)
                    // 2,
                ]
                d2udy2 = (1 / (self.dx[1] ** 2)) * d2udy2

                d2udz2 = torch.nn.functional.conv3d(
                    u, d2dz23D, stride=1, padding=0, bias=None
                )[
                    :,
                    :,
                    (self.d2dx21D.shape[0] - 1)
                    // 2 : -(self.d2dx21D.shape[0] - 1)
                    // 2,
                    (self.d2dx21D.shape[0] - 1)
                    // 2 : -(self.d2dx21D.shape[0] - 1)
                    // 2,
                    :,
                ]
                d2udz2 = (1 / (self.dx[2] ** 2)) * d2udz2

                return {
                    f"{self.invar}__x__x": d2udx2,
                    f"{self.invar}__y__y": d2udy2,
                    f"{self.invar}__z__z": d2udz2,
                }


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

        if self.order == 1:
            # inverse fourier transform out
            wx = torch.cat(
                [
                    torch.fft.ifftn(wx_h_i, dim=list(range(2, self.dim + 2))).real
                    for wx_h_i in wx_h
                ],
                dim=1,
            )

            if self.dim == 1:
                return {
                    f"{self.invar}__x": wx[:, 0:1],
                }
            elif self.dim == 2:
                return {
                    f"{self.invar}__x": wx[:, 0:1],
                    f"{self.invar}__y": wx[:, 1:2],
                }
            elif self.dim == 3:
                return {
                    f"{self.invar}__x": wx[:, 0:1],
                    f"{self.invar}__y": wx[:, 1:2],
                    f"{self.invar}__z": wx[:, 2:3],
                }
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

            if self.dim == 1:
                return {
                    f"{self.invar}__x__x": wxx[:, 0:1],
                }
            elif self.dim == 2:
                return {
                    f"{self.invar}__x__x": wxx[:, 0:1],
                    f"{self.invar}__y__y": wxx[:, 1:2],
                }
            elif self.dim == 3:
                return {
                    f"{self.invar}__x__x": wxx[:, 0:1],
                    f"{self.invar}__y__y": wxx[:, 1:2],
                    f"{self.invar}__z__z": wxx[:, 2:3],
                }


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
    def __init__(self):
        self.methods = {}
        self._register_methods()

    def _register_methods(self):
        self.methods["autodiff"] = GradientsAutoDiff
        self.methods["meshless_finite_difference"] = GradientsMeshlessFiniteDifference
        self.methods["finite_difference"] = GradientsFiniteDifference
        self.methods["spectral"] = GradientsSpectral
        self.methods["least_squares"] = GradientsLeastSquares

    def get_gradient_module(self, method_name, invar, **kwargs):
        return self.methods[method_name](invar, **kwargs)

    def compute_gradients(self, input_dict, method_name=None, invar=None, **kwargs):
        method = self.methods[method_name](invar, **kwargs)
        return method.forward(input_dict)

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


if __name__ == "__main__":

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return (
                torch.sin(x[:, 0:1])
                + torch.sin(8 * x[:, 1:2])
                + torch.sin(4 * x[:, 2:3])
            )

    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True)
    y = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True)
    z = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True)

    # Connectivity information
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

    # Convert connectivity to tensor
    edges = torch.tensor(edges)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)
    coords_unstructured = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    node_ids = torch.arange(coords_unstructured.size(0)).reshape(-1, 1)

    model = Model()

    # instantiate gradient calculator
    grad_calc = GradientCalculator()

    # compute grads using autodiff
    input_dict = {"coordinates": coords_unstructured, "u": model(coords_unstructured)}
    grad_u_autodiff = grad_calc.compute_gradients(
        input_dict, method_name="autodiff", invar="u"
    )
    print("AutoDiff: ", grad_u_autodiff.keys())

    # compute grads using meshless fd
    po_posx, po_negx, po_posy, po_negy, po_posz, po_negz = compute_stencil(
        coords_unstructured, model, dx=0.001
    )
    input_dict = {
        "u": model(coords_unstructured),
        "u>>x::1": po_posx,
        "u>>x::-1": po_negx,
        "u>>y::1": po_posy,
        "u>>y::-1": po_negy,
        "u>>z::1": po_posz,
        "u>>z::-1": po_negz,
    }
    grads_u_meshless_fd = grad_calc.compute_gradients(
        input_dict, method_name="meshless_finite_difference", invar="u", dx=0.001
    )
    print("MFD: ", grads_u_meshless_fd.keys())

    # compute grads using finite difference
    input_dict = {"u": model(coords)}
    grads_u_fd = grad_calc.compute_gradients(
        input_dict, method_name="finite_difference", invar="u", dx=[0.001, 0.001, 0.001]
    )
    print("FD: ", grads_u_fd.keys())

    # compute grads using spectral derivatives
    input_dict = {"u": model(coords)}
    grads_u_spectral = grad_calc.compute_gradients(
        input_dict,
        method_name="spectral",
        invar="u",
        ell=[2 * np.pi, 2 * np.pi, 2 * np.pi],
    )
    print("Spectral: ", grads_u_spectral.keys())

    # compute grads using least squares method
    input_dict = {
        "u": model(coords_unstructured),
        "coordinates": coords_unstructured,
        "nodes": node_ids,
        "edges": edges,
        "connectivity_tensor": compute_connectivity_tensor(
            coords_unstructured, node_ids, edges
        ),
    }
    grads_u_ls = grad_calc.compute_gradients(
        input_dict, method_name="least_squares", invar="u"
    )
    print("Least Squares: ", grads_u_ls.keys())
