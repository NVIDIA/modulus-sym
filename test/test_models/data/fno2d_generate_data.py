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
import operator
from functools import reduce
from functools import partial

paddle.seed(seed=0)
np.random.seed(0)
cuda_device = str("cpu:0").replace("cuda", "gpu")


class SpectralConv2d(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        out_35 = paddle.create_parameter(
            shape=(
                self.scale
                * paddle.rand(
                    shape=[in_channels, out_channels, self.modes1, self.modes2],
                    dtype="complex64",
                )
            ).shape,
            dtype=(
                self.scale
                * paddle.rand(
                    shape=[in_channels, out_channels, self.modes1, self.modes2],
                    dtype="complex64",
                )
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                self.scale
                * paddle.rand(
                    shape=[in_channels, out_channels, self.modes1, self.modes2],
                    dtype="complex64",
                )
            ),
        )
        out_35.stop_gradient = not True
        self.weights1 = out_35
        out_36 = paddle.create_parameter(
            shape=(
                self.scale
                * paddle.rand(
                    shape=[in_channels, out_channels, self.modes1, self.modes2],
                    dtype="complex64",
                )
            ).shape,
            dtype=(
                self.scale
                * paddle.rand(
                    shape=[in_channels, out_channels, self.modes1, self.modes2],
                    dtype="complex64",
                )
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                self.scale
                * paddle.rand(
                    shape=[in_channels, out_channels, self.modes1, self.modes2],
                    dtype="complex64",
                )
            ),
        )
        out_36.stop_gradient = not True
        self.weights2 = out_36

    def compl_mul2d(self, input, weights):
        return paddle.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = paddle.fft.rfft2(x=x)
        out_ft = paddle.zeros(
            shape=[batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1],
            dtype="complex64",
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        x = paddle.fft.irfft2(x=out_ft, s=(x.shape[-2], x.shape[-1]))
        return x


class FNO2d(paddle.nn.Layer):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9
        self.fc0 = paddle.nn.Linear(in_features=3, out_features=self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = paddle.nn.Conv2D(
            in_channels=self.width, out_channels=self.width, kernel_size=1
        )
        self.w1 = paddle.nn.Conv2D(
            in_channels=self.width, out_channels=self.width, kernel_size=1
        )
        self.w2 = paddle.nn.Conv2D(
            in_channels=self.width, out_channels=self.width, kernel_size=1
        )
        self.w3 = paddle.nn.Conv2D(
            in_channels=self.width, out_channels=self.width, kernel_size=1
        )
        self.fc1 = paddle.nn.Linear(in_features=self.width, out_features=128)
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        batchsize = x.shape[0]
        grid = self.get_grid(x.shape, x.place)
        x = paddle.concat(x=(x, grid), axis=-1)
        x = self.fc0(x)
        x = x.transpose(perm=[0, 3, 1, 2])
        x = paddle.nn.functional.pad(x, [0, self.padding, 0, self.padding])
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = paddle.nn.functional.gelu(x=x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = paddle.nn.functional.gelu(x=x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = paddle.nn.functional.gelu(x=x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = x[..., : -self.padding, : -self.padding]
        x = x.transpose(perm=[0, 2, 3, 1])
        x = self.fc1(x)
        x = paddle.nn.functional.gelu(x=x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = paddle.to_tensor(data=np.linspace(0, 1, size_x), dtype="float32")
        gridx = gridx.reshape(1, size_x, 1, 1).tile(
            repeat_times=[batchsize, 1, size_y, 1]
        )
        gridy = paddle.to_tensor(data=np.linspace(0, 1, size_y), dtype="float32")
        gridy = gridy.reshape(1, 1, size_y, 1).tile(
            repeat_times=[batchsize, size_x, 1, 1]
        )
        return paddle.concat(x=(gridx, gridy), axis=-1).to(device)


modes = 12
width = 32
model = FNO2d(modes, modes, width).to(cuda_device)
x_numpy = np.random.rand(100, 50, 50, 1).astype(np.float32)
x_tensor = paddle.to_tensor(data=x_numpy).to(cuda_device)
y_tensor = model(x_tensor)
y_numpy = y_tensor.detach().numpy()
Wbs = {
    _name: _value.data.detach().numpy() for _name, _value in model.named_parameters()
}
params = {"modes": [modes, modes], "width": width, "padding": [9, 9]}
np.savez_compressed(
    "test_fno2d.npz", data_in=x_numpy, data_out=y_numpy, params=params, Wbs=Wbs
)
