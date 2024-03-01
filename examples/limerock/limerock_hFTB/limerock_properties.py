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

#############
# Real Params
#############
# fluid params
fluid_viscosity = 1.84e-05  # kg/m-s
fluid_density = 1.1614  # kg/m3
fluid_specific_heat = 1005  # J/(kg K)
fluid_conductivity = 0.0261  # W/(m K)

# copper params
copper_density = 8930  # kg/m3
copper_specific_heat = 385  # J/(kg K)
copper_conductivity = 385  # W/(m K)

# boundary params
inlet_velocity = 5.7  # m/s
inlet_temp = 0  # K

# source
source_term = 2127.71  # K/m
source_origin = (-0.061667, -0.15833, limerock.geo_bounds_lower[2])
source_dim = (0.1285, 0.31667, 0)

################
# Non dim params
################
length_scale = 0.0575  # m
velocity_scale = 5.7  # m/s
time_scale = length_scale / velocity_scale  # s
density_scale = 1.1614  # kg/m3
mass_scale = density_scale * length_scale**3  # kg
pressure_scale = mass_scale / (length_scale * time_scale**2)  # kg / (m s**2)
temp_scale = 273.15  # K
watt_scale = (mass_scale * length_scale**2) / (time_scale**3)  # kg m**2 / s**3
joule_scale = (mass_scale * length_scale**2) / (time_scale**2)  # kg * m**2 / s**2

##############################
# Nondimensionalization Params
##############################
# fluid params
nd_fluid_viscosity = fluid_viscosity / (
    length_scale**2 / time_scale
)  # need to divide by density to get previous viscosity
nd_fluid_density = fluid_density / density_scale
nd_fluid_specific_heat = fluid_specific_heat / (joule_scale / (mass_scale * temp_scale))
nd_fluid_conductivity = fluid_conductivity / (watt_scale / (length_scale * temp_scale))
nd_fluid_diffusivity = nd_fluid_conductivity / (
    nd_fluid_specific_heat * nd_fluid_density
)

# copper params
nd_copper_density = copper_density / (mass_scale / length_scale**3)
nd_copper_specific_heat = copper_specific_heat / (
    joule_scale / (mass_scale * temp_scale)
)
nd_copper_conductivity = copper_conductivity / (
    watt_scale / (length_scale * temp_scale)
)
nd_copper_diffusivity = nd_copper_conductivity / (
    nd_copper_specific_heat * nd_copper_density
)

# boundary params
nd_inlet_velocity = inlet_velocity / velocity_scale
nd_volumetric_flow = limerock.inlet_area * nd_inlet_velocity
nd_inlet_temp = inlet_temp / temp_scale
nd_source_term = source_term / (temp_scale / length_scale)
