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
from typing import Union, List


class FirstDerivO2(torch.nn.Module):
    def __init__(self, dim: int, dx: Union[float, List[float]]):
        super(FirstDerivO2, self).__init__()
        self.dim = dim
        if isinstance(dx, float):
            dx = [dx for _ in range(dim)]
        self.dx = dx

        self.register_buffer("ddx1D", torch.Tensor([-1.0 / 2.0, 0.0, 1.0 / 2.0]))

    def get_convolution_kernel(self, axis):
        shape = [1, 1] + axis * [1] + [-1] + (self.dim - axis - 1) * [1]
        return torch.reshape(self.ddx1D, shape)

    def apply_convolution(self, u, kernel, axis):
        conv_result = torch.nn.functional.conv3d(
            u, kernel, stride=1, padding=0, bias=None
        )
        slice_index = (kernel.flatten().shape[-1] - 1) // 2
        if self.dim == 1:
            return conv_result
        elif self.dim == 2:
            if axis == 0:
                return conv_result[:, :, :, slice_index:-slice_index]
            elif axis == 1:
                return conv_result[:, :, slice_index:-slice_index, :]
        elif self.dim == 3:
            if axis == 0:
                return conv_result[
                    :, :, :, slice_index:-slice_index, slice_index:-slice_index
                ]
            elif axis == 1:
                return conv_result[
                    :, :, slice_index:-slice_index, :, slice_index:-slice_index
                ]
            elif axis == 2:
                return conv_result[
                    :, :, slice_index:-slice_index, slice_index:-slice_index, :
                ]

    def forward(self, u):
        u = torch.nn.functional.pad(u, self.dim * (1, 1), "replicate")
        result = []
        for axis in range(self.dim):
            kernel = self.get_convolution_kernel(axis)
            conv_result = self.apply_convolution(u, kernel, axis)
            conv_result = (1 / self.dx[axis]) * conv_result
            result.append(conv_result)
        return result


class SecondDerivO2(torch.nn.Module):
    def __init__(self, dim: int, dx: Union[float, List[float]]):
        super(SecondDerivO2, self).__init__()
        self.dim = dim
        if isinstance(dx, float):
            dx = [dx for _ in range(dim)]
        self.dx = dx

        self.register_buffer("d2dx21D", torch.Tensor([1.0, -2.0, 1.0]))

    def get_convolution_kernel(self, axis):
        shape = [1, 1] + axis * [1] + [-1] + (self.dim - axis - 1) * [1]
        return torch.reshape(self.d2dx21D, shape)

    def apply_convolution(self, u, kernel, axis):
        conv_result = torch.nn.functional.conv3d(
            u, kernel, stride=1, padding=0, bias=None
        )
        slice_index = (kernel.shape[-1] - 1) // 2
        if self.dim == 1:
            return conv_result
        elif self.dim == 2:
            if axis == 0:
                return conv_result[:, :, :, slice_index:-slice_index]
            elif axis == 1:
                return conv_result[:, :, slice_index:-slice_index, :]
        elif self.dim == 3:
            if axis == 0:
                return conv_result[
                    :, :, :, slice_index:-slice_index, slice_index:-slice_index
                ]
            elif axis == 1:
                return conv_result[
                    :, :, slice_index:-slice_index, :, slice_index:-slice_index
                ]
            elif axis == 2:
                return conv_result[
                    :, :, slice_index:-slice_index, slice_index:-slice_index, :
                ]

    def forward(self, u):
        u = torch.nn.functional.pad(u, self.dim * (1, 1), "replicate")
        result = []
        for axis in range(self.dim):
            kernel = self.get_convolution_kernel(axis)
            conv_result = self.apply_convolution(u, kernel, axis)
            conv_result = (1 / (self.dx[axis] ** 2)) * conv_result
            result.append(conv_result)
        return result
