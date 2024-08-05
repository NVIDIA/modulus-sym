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


import sympy.core
from sympy import Symbol, Function, Number

from modulus.sym.eq.pde import PDE


class BoxPDE(PDE):
    """
    The Schrödinger equation for a particle in a box.
    Zero potential inside the box, infinite potential outside.

    Also includes terms intended to prevent the eigenvalue from becoming dropping to zero:
    1. A 1/E term in the loss function
    2. A driving term that forces the eigenvalue above some constant value
     (can be set in config, 0 by default).
    """

    name = "BoxPDE"

    def __init__(self, hbar=1.0, mass=1.0, e_const=0.0):
        self.hbar = hbar
        self.mass = mass
        self.e_const = e_const
        self.equations = {}
        self._build_equations()

    def _build_equations(self):
        x = Symbol("x")
        E = Symbol("E")

        input_variables = {"x": x}
        psi = Function("psi")(*input_variables)

        self.equations["psi"] = psi
        self.equations["E"] = E
        self.equations["E_inv"] = 1.0 / (0.001 + sympy.Abs(E))

        # Main Schrödinger equation.
        p_prefix = Number(-0.5 * self.hbar**2 / self.mass)
        self.equations["sch_equation"] = p_prefix * psi.diff(x, x) - E * psi

        # Calculate the squared-magnitude of the wave function
        # We set a constraint to force this to be 1.
        self.equations["psi_norm"] = psi * sympy.conjugate(psi)

        # L_drive loss term for soft-setting a minimum eigenvalue.
        self.equations["L_drive"] = sympy.exp(-E + self.e_const)
