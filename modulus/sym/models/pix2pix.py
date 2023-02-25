""""""
"""
Pix2Pix model. This code was modified from, https://github.com/NVIDIA/pix2pixHD

The following license is provided from their source,

Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
BSD License. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


--------------------------- LICENSE FOR pytorch-CycleGAN-and-pix2pix ----------------
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import functools
from typing import List, Dict
from torch.autograd import Variable
import numpy as np

from modulus.sym.key import Key
import modulus.sym.models.layers as layers
from modulus.sym.models.layers import Activation
from modulus.sym.models.arch import Arch

Tensor = torch.Tensor


class Pix2PixModelCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        conv_layer_size: int = 64,
        n_downsampling: int = 3,
        n_upsampling: int = 3,
        n_blocks: int = 3,
        batch_norm: bool = False,
        padding_type: str = "reflect",
        activation_fn: Activation = Activation.RELU,
    ):
        assert (
            n_blocks >= 0 and n_downsampling >= 0 and n_upsampling >= 0
        ), "Invalid arch params"
        assert padding_type in ["reflect", "zero", "replicate"], "Invalid padding type"
        super().__init__()

        activation = layers.get_activation_fn(activation_fn, module=True, inplace=True)
        # set padding and convolutions
        if dimension == 1:
            padding = nn.ReflectionPad1d(3)
            conv = nn.Conv1d
            trans_conv = nn.ConvTranspose1d
            norm = nn.BatchNorm1d
        elif dimension == 2:
            padding = nn.ReflectionPad2d(3)
            conv = nn.Conv2d
            trans_conv = nn.ConvTranspose2d
            norm = nn.BatchNorm2d
        elif dimension == 3:
            padding = nn.ReflectionPad3d(3)
            conv = nn.Conv3d
            trans_conv = nn.ConvTranspose3d
            norm = nn.BatchNorm3d
        else:
            raise ValueError(
                f"Pix2Pix only supported dimensions 1, 2, 3. Got {dimension}"
            )

        model = [
            padding,
            conv(in_channels, conv_layer_size, kernel_size=7, padding=0),
        ]
        if batch_norm:
            model.append(norm(conv_layer_size))
        model.append(activation)

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model.append(
                conv(
                    conv_layer_size * mult,
                    conv_layer_size * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            if batch_norm:
                model.append(norm(conv_layer_size * mult * 2))
            model.append(activation)

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    dimension,
                    conv_layer_size * mult,
                    padding_type=padding_type,
                    activation=activation,
                    use_batch_norm=batch_norm,
                )
            ]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model.append(
                trans_conv(
                    int(conv_layer_size * mult),
                    int(conv_layer_size * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            if batch_norm:
                model.append(norm(int(conv_layer_size * mult / 2)))
            model.append(activation)

        # super-resolution layers
        for i in range(max([0, n_upsampling - n_downsampling])):
            model.append(
                trans_conv(
                    int(conv_layer_size),
                    int(conv_layer_size),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            if batch_norm:
                model.append(norm(conv_layer_size))
            model.append(activation)

        model += [
            padding,
            conv(conv_layer_size, out_channels, kernel_size=7, padding=0),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input: Tensor) -> Tensor:
        y = self.model(input)
        return y


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        channels: int,
        padding_type: str = "zero",
        activation: nn.Module = nn.ReLU(True),
        use_batch_norm: bool = False,
        use_dropout: bool = False,
    ):
        super().__init__()

        if dimension == 1:
            conv = nn.Conv1d
            if padding_type == "reflect":
                padding = nn.ReflectionPad1d(1)
            elif padding_type == "replicate":
                padding = nn.ReplicationPad1d(1)
            elif padding_type == "zero":
                padding = 1
            norm = nn.BatchNorm1d
        elif dimension == 2:
            conv = nn.Conv2d
            if padding_type == "reflect":
                padding = nn.ReflectionPad2d(1)
            elif padding_type == "replicate":
                padding = nn.ReplicationPad2d(1)
            elif padding_type == "zero":
                padding = 1
            norm = nn.BatchNorm2d
        elif dimension == 3:
            conv = nn.Conv3d
            if padding_type == "reflect":
                padding = nn.ReflectionPad3d(1)
            elif padding_type == "replicate":
                padding = nn.ReplicationPad3d(1)
            elif padding_type == "zero":
                padding = 1
            norm = nn.BatchNorm3d

        conv_block = []
        p = 0
        if padding_type != "zero":
            conv_block += [padding]

        conv_block.append(conv(channels, channels, kernel_size=3, padding=p))
        if use_batch_norm:
            conv_block.append(norm(channels))
        conv_block.append(activation)

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        if padding_type != "zero":
            conv_block += [padding]
        conv_block += [
            conv(channels, channels, kernel_size=3, padding=p),
        ]
        if use_batch_norm:
            conv_block.append(norm(channels))

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: Tensor) -> Tensor:
        out = x + self.conv_block(x)
        return out


class Pix2PixArch(Arch):
    """Convolutional encoder-decoder based on pix2pix generator models.

    Note
    ----
    The pix2pix architecture supports options for 1D, 2D and 3D fields which can
    be constroled using the `dimension` parameter.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list. The key dimension size should equal the variables channel dim.
    output_keys : List[Key]
        Output key list. The key dimension size should equal the variables channel dim.
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    conv_layer_size : int, optional
        Latent channel size after first convolution, by default 64
    n_downsampling : int, optional
        Number of downsampling/upsampling blocks, by default 3
    n_blocks : int, optional
        Number of residual blocks in middle of model, by default 3
    scaling_factor : int, optional
        Scaling factor to increase the output feature size compared to the input
        (1, 2, 4, or 8), by default 1
    activation_fn : Activation, optional
        Activation function, by default :obj:`Activation.RELU`
    batch_norm : bool, optional
        Batch normalization, by default False
    padding_type : str, optional
        Padding type ('constant', 'reflect', 'replicate' or 'circular'),
        by default "reflect"

    Variable Shape
    --------------
    Input variable tensor shape:

    - 1D: :math:`[N, size, W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Output variable tensor shape:

    - 1D: :math:`[N, size,  W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Note
    ----
    Reference:  Isola, Phillip, et al. “Image-To-Image translation with conditional
    adversarial networks” Conference on Computer Vision and Pattern Recognition, 2017.
    https://arxiv.org/abs/1611.07004

    Reference: Wang, Ting-Chun, et al. “High-Resolution image synthesis and semantic
    manipulation with conditional GANs” Conference on Computer Vision and Pattern
    Recognition, 2018. https://arxiv.org/abs/1711.11585

    Note
    ----
    Based on the implementation: https://github.com/NVIDIA/pix2pixHD
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        dimension: int,
        detach_keys: List[Key] = [],
        conv_layer_size: int = 64,
        n_downsampling: int = 3,
        n_blocks: int = 3,
        scaling_factor: int = 1,
        activation_fn: Activation = Activation.RELU,
        batch_norm: bool = False,
        padding_type="reflect",
    ):
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )
        in_channels = sum(self.input_key_dict.values())
        out_channels = sum(self.output_key_dict.values())
        self.var_dim = 1

        # Scaling factor must be 1, 2, 4, or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {
            1,
            2,
            4,
            8,
        }, "The scaling factor must be 1, 2, 4, or 8!"
        n_upsampling = n_downsampling + int(np.log2(scaling_factor))

        self._impl = Pix2PixModelCore(
            in_channels,
            out_channels,
            dimension,
            conv_layer_size,
            n_downsampling,
            n_upsampling,
            n_blocks,
            batch_norm,
            padding_type,
            activation_fn,
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=1,
            input_scales=self.input_scales,
        )
        output = self._impl(input)
        return self.prepare_output(
            output, self.output_key_dict, dim=1, output_scales=self.output_scales
        )
