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

import numpy as np
import matplotlib.pyplot as plt

base_dir = "outputs/spring_mass_solver/validators/"

# plot in 1d
data = np.load(base_dir + "validator.npz", allow_pickle=True)
data = np.atleast_1d(data.f.arr_0)[0]

plt.plot(data["t"], data["true_x1"], label="True x1")
plt.plot(data["t"], data["true_x2"], label="True x2")
plt.plot(data["t"], data["true_x3"], label="True x3")
plt.plot(data["t"], data["pred_x1"], label="Pred x1")
plt.plot(data["t"], data["pred_x2"], label="Pred x2")
plt.plot(data["t"], data["pred_x3"], label="Pred x3")
plt.legend()
plt.savefig("comparison.png")
