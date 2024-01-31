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
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags


def Laplacian_1D_eig(a, b, N, eps=lambda x: np.ones_like(x), k=3):
    n = N - 2
    h = (b - a) / (N - 1)

    L = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    L = -L / h**2

    x = np.linspace(a, b, num=N)
    M = diags([eps(x[1:-1])], [0])

    eigvals, eigvecs = eigsh(L, k=k, M=M, which="SM")
    eigvecs = np.vstack((np.zeros((1, k)), eigvecs, np.zeros((1, k))))
    norm_eigvecs = np.linalg.norm(eigvecs, axis=0)
    eigvecs /= norm_eigvecs
    return eigvals.astype(np.float32), eigvecs.astype(np.float32), x.astype(np.float32)
