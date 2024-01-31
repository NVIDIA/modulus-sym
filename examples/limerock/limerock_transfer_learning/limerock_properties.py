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

from limerock_geometry import LimeRock

# make limerock
limerock = LimeRock()

# Real Params
# fluid params
fluid_viscosity = 1.84e-05  # kg/m-s
fluid_density = 1.1614  # kg/m3
fluid_specific_heat = 1005  # J/kg-K
fluid_conductivity = 0.0261  # W/m-K

# copper params
copper_density = 8930  # kg/m3
copper_specific_heat = 385  # J/kg-K
copper_conductivity = 385  # W/m-K

# boundary params
length_scale = 0.0575  # m
inlet_velocity = 5.7  # m/s
inlet_velocity_normalized = 1.0
power = 120  # W
ambient_temperature = 61  # degree Celsius

# Nondimensionalization Params
# fluid params
nu = limerock.scale * fluid_viscosity / (fluid_density * inlet_velocity)
rho = 1.0
volumetric_flow = limerock.inlet_area * inlet_velocity_normalized

# heat params
D_solid = 0.10
D_fluid = 0.02
source_grad = 1.5
source_area = 0.25**2
source_origin = (-0.061667, -0.15833, limerock.geo_bounds_lower[2])
source_dim = (0.1285, 0.31667, 0)
