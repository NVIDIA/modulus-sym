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

from sympy import Symbol, Eq, tanh
import numpy as np

from modulus.sym.geometry.primitives_3d import Box, Channel, Plane
from modulus.sym.geometry import Parameterization, Parameter

# geometry params for domain
channel_origin = (-2.5, -0.5, -0.5625)
channel_dim = (5.0, 1.0, 1.125)
heat_sink_base_origin = (-0.75, -0.5, -0.4375)
heat_sink_base_dim = (0.65, 0.05, 0.875)
fin_origin = heat_sink_base_origin
fin_dim = (0.65, 0.8625, 0.0075)
total_fins = 17
flow_box_origin = (-0.85, -0.5, -0.5625)
flow_box_dim = (0.85, 1.0, 1.125)
source_origin = (-0.55, -0.5, -0.125)
source_dim = (0.25, 0.0, 0.25)

# define sympy varaibles to parametize domain curves
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

# define geometry
# channel
channel_base = Channel(
    channel_origin,
    (
        channel_origin[0] + channel_dim[0],
        channel_origin[1] + channel_dim[1],
        channel_origin[2] + channel_dim[2],
    ),
)
channel_void = Channel(
    (channel_origin[0], channel_origin[1], channel_origin[2] + channel_dim[2] / 2),
    (
        channel_origin[0] + channel_dim[0],
        channel_origin[1] + channel_dim[1],
        channel_origin[2] + channel_dim[2],
    ),
)
channel = channel_base - channel_void


# fpga heat sink
heat_sink_base = Box(
    heat_sink_base_origin,
    (
        heat_sink_base_origin[0] + heat_sink_base_dim[0],  # base of heat sink
        heat_sink_base_origin[1] + heat_sink_base_dim[1],
        heat_sink_base_origin[2] + heat_sink_base_dim[2] / 2,
    ),
)
fin_center = (
    fin_origin[0] + fin_dim[0] / 2,
    fin_origin[1] + fin_dim[1] / 2,
    fin_origin[2] + fin_dim[2] / 2,
)
fin = Box(
    fin_origin,
    (
        fin_origin[0] + fin_dim[0],
        fin_origin[1] + fin_dim[1],
        fin_origin[2] + fin_dim[2],
    ),
)
gap = (heat_sink_base_dim[2] - fin_dim[2]) / (total_fins - 1)  # gap between fins
fin = fin.repeat(
    gap,
    repeat_lower=(0, 0, 0),
    repeat_higher=(0, 0, total_fins - 1),
    center=fin_center,
)

fin_void = Box(
    (channel_origin[0], channel_origin[1], channel_origin[2] + channel_dim[2] / 2),
    (
        channel_origin[0] + channel_dim[0],
        channel_origin[1] + channel_dim[1],
        channel_origin[2] + channel_dim[2],
    ),
)

fin = fin - fin_void

fpga = heat_sink_base + fin

# entire geometry
geo = channel - fpga

# inlet and outlet
inlet = Plane(
    channel_origin,
    (
        channel_origin[0],
        channel_origin[1] + channel_dim[1],
        channel_origin[2] + channel_dim[2] / 2,
    ),
    -1,
)
outlet = Plane(
    (channel_origin[0] + channel_dim[0], channel_origin[1], channel_origin[2]),
    (
        channel_origin[0] + channel_dim[0],
        channel_origin[1] + channel_dim[1],
        channel_origin[2] + channel_dim[2] / 2,
    ),
    1,
)

# planes for integral continuity
x_pos = Parameter("x_pos")
integral_plane = Plane(
    (x_pos, channel_origin[1], channel_origin[2]),
    (x_pos, channel_origin[1] + channel_dim[1], channel_origin[2] + channel_dim[2]),
    1,
    parameterization=Parameterization({x_pos: (-0.75, 0.0)}),
)
