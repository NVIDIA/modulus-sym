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

# import libraries
import numpy as np
import chaospy

# define parameter ranges
fin_front_top_cut_angle_ranges = (0.0, np.pi / 6.0)
fin_front_bottom_cut_angle_ranges = (0.0, np.pi / 6.0)
fin_back_top_cut_angle_ranges = (0.0, np.pi / 6.0)
fin_back_bottom_cut_angle_ranges = (0.0, np.pi / 6.0)

# generate samples
samples = chaospy.generate_samples(
    order=30,
    domain=np.array(
        [
            fin_front_top_cut_angle_ranges,
            fin_front_bottom_cut_angle_ranges,
            fin_back_top_cut_angle_ranges,
            fin_back_bottom_cut_angle_ranges,
        ]
    ).T,
    rule="halton",
)
samples = samples.T
np.random.shuffle(samples)
np.savetxt("samples.txt", samples)
