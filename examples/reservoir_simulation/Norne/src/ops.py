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
import torch.nn.functional as F


def dx(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor"

    var = inpt[:, channel : channel + 1, :, :]

    # get filter
    if order == 1:
        ddx1D = torch.Tensor(
            [
                -0.5,
                0.0,
                0.5,
            ]
        ).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor(
            [
                -1.0 / 60.0,
                3.0 / 20.0,
                -3.0 / 4.0,
                0.0,
                3.0 / 4.0,
                -3.0 / 20.0,
                1.0 / 60.0,
            ]
        ).to(inpt.device)
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])

    # apply convolution
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]

    return output


def ddx(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute second order numerical derivatives of input tensor"

    var = inpt[:, channel : channel + 1, :, :]

    # get filter
    if order == 1:
        ddx1D = torch.Tensor(
            [
                1.0,
                -2.0,
                1.0,
            ]
        ).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor(
            [
                1.0 / 90.0,
                -3.0 / 20.0,
                3.0 / 2.0,
                -49.0 / 18.0,
                3.0 / 2.0,
                -3.0 / 20.0,
                1.0 / 90.0,
            ]
        ).to(inpt.device)
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])

    # apply convolution
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx**2) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]

    return output


def compute_differential(u, dxf):
    # Assuming u has shape: [batch_size, channels, nz, height, width]
    # Aim is to compute derivatives along height, width, and nz for each slice in nz

    batch_size, channels, nz, height, width = u.shape
    derivatives_x = []
    derivatives_y = []
    derivatives_z = []  # List to store derivatives in z direction

    for i in range(nz):
        slice_u = u[:, :, i, :, :]  # shape: [batch_size, channels, height, width]

        # Compute derivatives for this slice
        dudx_fdm = dx(slice_u, dx=dxf, channel=0, dim=0, order=1, padding="replication")
        dudy_fdm = dx(slice_u, dx=dxf, channel=0, dim=1, order=1, padding="replication")

        derivatives_x.append(dudx_fdm)
        derivatives_y.append(dudy_fdm)

        # Compute the derivative in z direction
        # Avoid the boundaries of the volume in z direction
        if i > 0 and i < nz - 1:
            dudz_fdm = (u[:, :, i + 1, :, :] - u[:, :, i - 1, :, :]) / (2 * dxf)
            derivatives_z.append(dudz_fdm)
        else:
            # This handles the boundaries where the derivative might not be well-defined
            # Depending on your application, you can either use forward/backward differences or pad with zeros or replicate values
            # Here, as an example, I'm padding with zeros
            dudz_fdm = torch.zeros_like(slice_u)
            derivatives_z.append(dudz_fdm)

    # Stack results to get tensors of shape [batch_size, channels, nz, height, width]
    dudx_fdm = torch.stack(derivatives_x, dim=2)
    dudy_fdm = torch.stack(derivatives_y, dim=2)
    dudz_fdm = torch.stack(derivatives_z, dim=2)  # Stack the z derivatives

    return dudx_fdm, dudy_fdm, dudz_fdm  # Return the z derivatives as well


def compute_second_differential(u, dxf):
    """Computes the x, y, and z second derivatives for each slice in the nz dimension of tensor u."""

    batch_size, channels, nz, height, width = u.shape
    second_derivatives_x = []
    second_derivatives_y = []
    second_derivatives_z = []  # List to store second derivatives in z direction

    for i in range(nz):
        slice_u = u[:, :, i, :, :]  # Extract the ith slice in the nz dimension

        # Compute second derivatives for this slice in x and y
        dduddx_fdm = ddx(
            slice_u, dx=dxf, channel=0, dim=0, order=1, padding="replication"
        )
        dduddy_fdm = ddx(
            slice_u, dx=dxf, channel=0, dim=1, order=1, padding="replication"
        )

        second_derivatives_x.append(dduddx_fdm)
        second_derivatives_y.append(dduddy_fdm)

        # Compute the second derivative in z direction
        # Avoid the boundaries of the volume in z direction
        if i > 1 and i < nz - 2:
            dduddz_fdm = (u[:, :, i + 2, :, :] - 2 * slice_u + u[:, :, i - 2, :, :]) / (
                dxf**2
            )
            second_derivatives_z.append(dduddz_fdm)
        else:
            # This handles the boundaries where the derivative might not be well-defined
            # Padding with zeros for simplicity. You may need to handle this differently based on your application
            dduddz_fdm = torch.zeros_like(slice_u)
            second_derivatives_z.append(dduddz_fdm)

    # Stack results along the nz dimension to get tensors of shape [batch_size, channels, nz, height, width]
    dduddx_fdm = torch.stack(second_derivatives_x, dim=2)
    dduddy_fdm = torch.stack(second_derivatives_y, dim=2)
    dduddz_fdm = torch.stack(
        second_derivatives_z, dim=2
    )  # Stack the z second derivatives

    return dduddx_fdm, dduddy_fdm, dduddz_fdm  # Return the z second derivatives as well


def compute_gradient_3d(inpt, dx, dim, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor for 3D data"

    # Define filter
    if order == 1:
        ddx1D = torch.Tensor([-0.5, 0.0, 0.5]).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor(
            [
                -1.0 / 60.0,
                3.0 / 20.0,
                -3.0 / 4.0,
                0.0,
                3.0 / 4.0,
                -3.0 / 20.0,
                1.0 / 60.0,
            ]
        ).to(inpt.device)

    # Reshape filter for 3D convolution
    padding_sizes = [(0, 0), (0, 0), (0, 0)]
    if dim == 0:
        ddx3D = ddx1D.view(1, 1, -1, 1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    elif dim == 1:
        ddx3D = ddx1D.view(1, 1, 1, -1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    else:  # dim == 2
        ddx3D = ddx1D.view(1, 1, 1, 1, -1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)

    # Iterate over channels and compute the gradient for each channel
    outputs = []
    for ch in range(inpt.shape[1]):
        channel_data = inpt[:, ch : ch + 1]
        if padding == "zeros":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "constant",
                0,
            )
        elif padding == "replication":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "replicate",
            )
        out_ch = F.conv3d(channel_data, ddx3D, padding=0) * (1.0 / dx)
        outputs.append(out_ch)

    # Stack results along the channel dimension
    output = torch.cat(outputs, dim=1)

    return output


def compute_second_order_gradient_3d(inpt, dx, dim, padding="zeros"):
    "Compute second order numerical derivatives (Laplacian) of input tensor for 3D data"

    # Define filter for second order derivative
    ddx1D = torch.Tensor([-1.0, 2.0, -1.0]).to(inpt.device)

    # Reshape filter for 3D convolution
    padding_sizes = [(0, 0), (0, 0), (0, 0)]
    if dim == 0:
        ddx3D = ddx1D.view(1, 1, -1, 1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    elif dim == 1:
        ddx3D = ddx1D.view(1, 1, 1, -1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    else:  # dim == 2
        ddx3D = ddx1D.view(1, 1, 1, 1, -1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)

    # Iterate over channels and compute the gradient for each channel
    outputs = []
    for ch in range(inpt.shape[1]):
        channel_data = inpt[:, ch : ch + 1]
        if padding == "zeros":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "constant",
                0,
            )
        elif padding == "replication":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "replicate",
            )
        out_ch = F.conv3d(channel_data, ddx3D, padding=0) * (1.0 / (dx**2))
        outputs.append(out_ch)

    # Stack results along the channel dimension
    output = torch.cat(outputs, dim=1)

    return output
