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

import torch
import numpy as np

from .functions import *
from modulus.sym.key import Key
from typing import Dict, List, Set, Optional, Union, Callable

Tensor = torch.Tensor


class FirstDerivO2(torch.nn.Module):
    def __init__(self, derivative_key: Key) -> None:
        super().__init__()
        assert (
            len(derivative_key.derivatives) == 1
        ), f"Key must have one derivative for first derivative calc"
        self.var = derivative_key.name
        self.indep_var = str(derivative_key.derivatives[0])
        self.out_name = str(derivative_key)

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        outputs[self.out_name] = FirstDerivO2_f.apply(
            inputs[f"{self.var}>>{self.indep_var}::1"],
            inputs[f"{self.var}>>{self.indep_var}::-1"],
            dx,
        )
        return outputs


class FirstDerivO4(torch.nn.Module):
    def __init__(self, derivative_key: Key) -> None:
        super().__init__()
        assert (
            len(derivative_key.derivatives) == 1
        ), f"Key must have one derivative for first derivative calc"
        self.var = derivative_key.name
        self.indep_var = str(derivative_key.derivatives[0])
        self.out_name = str(derivative_key)

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        outputs[self.out_name] = FirstDerivO4_f.apply(
            inputs[f"{self.var}>>{self.indep_var}::2"],
            inputs[f"{self.var}>>{self.indep_var}::1"],
            inputs[f"{self.var}>>{self.indep_var}::-1"],
            inputs[f"{self.var}>>{self.indep_var}::-2"],
            dx,
        )
        return outputs


class SecondDerivO2(torch.nn.Module):
    def __init__(self, derivative_key: Key) -> None:
        super().__init__()
        assert (
            len(derivative_key.derivatives) == 2
        ), f"Key must have two derivatives for second derivative calc"
        assert (
            derivative_key.derivatives[0] == derivative_key.derivatives[1]
        ), f"Derivatives keys should be the same"
        self.var = derivative_key.name
        self.indep_var = str(derivative_key.derivatives[0])
        self.out_name = str(derivative_key)

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        outputs[self.out_name] = SecondDerivO2_f.apply(
            inputs[f"{self.var}>>{self.indep_var}::1"],
            inputs[f"{self.var}"],
            inputs[f"{self.var}>>{self.indep_var}::-1"],
            dx,
        )
        return outputs


class SecondDerivO4(torch.nn.Module):
    def __init__(self, derivative_key: Key) -> None:
        super().__init__()
        assert (
            len(derivative_key.derivatives) == 2
        ), f"Key must have two derivatives for second derivative calc"
        assert (
            derivative_key.derivatives[0] == derivative_key.derivatives[1]
        ), f"Derivatives keys should be the same"
        self.var = derivative_key.name
        self.indep_var = str(derivative_key.derivatives[0])
        self.out_name = str(derivative_key)

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        outputs[self.out_name] = SecondDerivO4_f.apply(
            inputs[f"{self.var}>>{self.indep_var}::2"],
            inputs[f"{self.var}>>{self.indep_var}::1"],
            inputs[f"{self.var}"],
            inputs[f"{self.var}>>{self.indep_var}::-1"],
            inputs[f"{self.var}>>{self.indep_var}::-2"],
            dx,
        )
        return outputs


class MixedSecondDerivO2(torch.nn.Module):
    def __init__(self, derivative_key: Key) -> None:
        super().__init__()
        assert (
            len(derivative_key.derivatives) == 2
        ), f"Key must have two derivatives for second derivative calc"
        self.var = derivative_key.name
        self.indep_vars = [
            str(derivative_key.derivatives[0]),
            str(derivative_key.derivatives[1]),
        ]
        self.indep_vars.sort()
        self.out_name = str(derivative_key)

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        outputs[self.out_name] = MixedSecondDerivO2_f.apply(
            inputs[f"{self.var}>>{self.indep_vars[0]}::1&&{self.indep_vars[1]}::1"],
            inputs[f"{self.var}>>{self.indep_vars[0]}::-1&&{self.indep_vars[1]}::1"],
            inputs[f"{self.var}>>{self.indep_vars[0]}::1&&{self.indep_vars[1]}::-1"],
            inputs[f"{self.var}>>{self.indep_vars[0]}::-1&&{self.indep_vars[1]}::-1"],
            dx,
        )
        return outputs


class ThirdDerivO2(torch.nn.Module):
    def __init__(self, derivative_key: Key) -> None:
        super().__init__()
        assert (
            len(derivative_key.derivatives) == 3
        ), f"Key must have three derivatives for third derivative calc"
        assert (
            derivative_key.derivatives[0]
            == derivative_key.derivatives[1]
            == derivative_key.derivatives[2]
        ), f"Derivatives keys should be the same"
        self.var = derivative_key.name
        self.indep_var = str(derivative_key.derivatives[0])
        self.out_name = str(derivative_key)

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        outputs[self.out_name] = ThirdDerivO2_f.apply(
            inputs[f"{self.var}>>{self.indep_var}::2"],
            inputs[f"{self.var}>>{self.indep_var}::1"],
            inputs[f"{self.var}>>{self.indep_var}::-1"],
            inputs[f"{self.var}>>{self.indep_var}::-2"],
            dx,
        )
        return outputs


class ForthDerivO2(torch.nn.Module):
    def __init__(self, derivative_key: Key) -> None:
        super().__init__()
        assert (
            len(derivative_key.derivatives) == 4
        ), f"Key must have three derivatives for forth derivative calc"
        assert (
            derivative_key.derivatives[0]
            == derivative_key.derivatives[1]
            == derivative_key.derivatives[2]
            == derivative_key.derivatives[3]
        ), f"Derivatives keys should be the same"
        self.var = derivative_key.name
        self.indep_var = str(derivative_key.derivatives[0])
        self.out_name = str(derivative_key)
        self.register_buffer(
            "coeff",
            torch.Tensor([1.0, -4.0, 6.0, -4.0, 1.0]).double().unsqueeze(-1),
            persistent=False,
        )

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        outputs[self.out_name] = ForthDerivO2_f.apply(
            inputs[f"{self.var}>>{self.indep_var}::2"],
            inputs[f"{self.var}>>{self.indep_var}::1"],
            inputs[f"{self.var}"],
            inputs[f"{self.var}>>{self.indep_var}::-1"],
            inputs[f"{self.var}>>{self.indep_var}::-2"],
            dx,
        )
        return outputs


class DerivBase(torch.nn.Module):
    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__()
        self.derivative_keys = derivative_keys
        self.dx = dx
        self.order = order

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()

    @property
    def stencil(self) -> List[str]:
        """Returns list of stencil strings for this derivative

        Returns
        -------
        List[str]
            List of stencil strings

        Example
        -------
        Central 2nd derivative will return: `['x::1','x::0','x::-1']`
        """
        return list(self._stencil)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass that calculates the finite difference gradient

        Parameters
        ----------
        inputs : Dict[str, Tensor]
            Input tensor dictionary, should include points in FD stencil

        Returns
        -------
        Dict[str, Tensor]
            Output gradients
        """
        outputs = {}
        for module in self._eval:
            outputs.update(module(inputs, self.dx))
        return outputs


class FirstDeriv(DerivBase):
    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__(derivative_keys, dx, order, jit)
        assert (
            len(derivative_keys) == 0 or order == 2 or order == 4
        ), "Second and forth order first derivatives supported"
        for key in derivative_keys:
            assert (
                len(key.derivatives) == 1
            ), f"Key with {len(key.derivatives)} derivs supplied to first order deriv"

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()
        for key in self.derivative_keys:
            indep_vars = key.derivatives
            if order == 2:
                self._stencil = self._stencil.union(
                    {f"{indep_vars[0]}::-1", f"{indep_vars[0]}::1"}
                )
                eval_list.append(FirstDerivO2(key))
            elif order == 4:
                self._stencil = self._stencil.union(
                    {
                        f"{indep_vars[0]}::-2",
                        f"{indep_vars[0]}::-1",
                        f"{indep_vars[0]}::1",
                        f"{indep_vars[0]}::2",
                    }
                )
                eval_list.append(FirstDerivO4(key))

        self._eval = torch.nn.ModuleList(eval_list)


class SecondDeriv(DerivBase):
    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__(derivative_keys, dx, order, jit)
        assert (
            len(derivative_keys) == 0 or order == 2 or order == 4
        ), "Second and forth order second derivatives supported"
        for key in derivative_keys:
            assert (
                len(key.derivatives) == 2
            ), f"Key with {len(key.derivatives)} deriv keys supplied to second deriv"

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()
        for key in self.derivative_keys:
            indep_vars = key.derivatives
            if indep_vars[0] == indep_vars[1]:
                if order == 2:
                    self._stencil = self._stencil.union(
                        {
                            f"{indep_vars[0]}::-1",
                            f"{indep_vars[0]}::0",
                            f"{indep_vars[0]}::1",
                        }
                    )
                    eval_list.append(SecondDerivO2(key))
                elif order == 4:
                    self._stencil = self._stencil.union(
                        {
                            f"{indep_vars[0]}::-2",
                            f"{indep_vars[0]}::-1",
                            f"{indep_vars[0]}::0",
                            f"{indep_vars[0]}::1",
                            f"{indep_vars[0]}::2",
                        }
                    )
                    eval_list.append(SecondDerivO4(key))
            # Mixed derivative
            else:
                if order == 2:
                    indep_vars = [str(var) for var in indep_vars]
                    indep_vars.sort()  # Avoid redundent points like (z::-1&&y::1 and y::1&&z::-1)
                    self._stencil = self._stencil.union(
                        {
                            f"{indep_vars[0]}::-1&&{indep_vars[1]}::-1",
                            f"{indep_vars[0]}::1&&{indep_vars[1]}::-1",
                            f"{indep_vars[0]}::-1&&{indep_vars[1]}::1",
                            f"{indep_vars[0]}::1&&{indep_vars[1]}::1",
                        }
                    )
                    eval_list.append(MixedSecondDerivO2(key))
                elif order == 4:
                    raise NotImplementedError(
                        "Fourth order mixed second derivatives not supported"
                    )

        self._eval = torch.nn.ModuleList(eval_list)


class ThirdDeriv(DerivBase):
    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__(derivative_keys, dx, order, jit)
        assert (
            len(derivative_keys) == 0 or order == 2
        ), "Second order third derivatives supported"
        for key in derivative_keys:
            assert (
                len(key.derivatives) == 3
            ), f"Key with {len(key.derivatives)} deriv keys supplied to third deriv"
            assert (
                key.derivatives[0] == key.derivatives[1] == key.derivatives[2]
            ), f"Mixed third derivatives not supported"

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()
        for key in self.derivative_keys:
            indep_vars = key.derivatives
            if order == 2:
                self._stencil = self._stencil.union(
                    {
                        f"{indep_vars[0]}::-2",
                        f"{indep_vars[0]}::-1",
                        f"{indep_vars[0]}::1",
                        f"{indep_vars[0]}::2",
                    }
                )
                eval_list.append(ThirdDerivO2(key))

        self._eval = torch.nn.ModuleList(eval_list)


class ForthDeriv(DerivBase):
    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__(derivative_keys, dx, order, jit)
        assert (
            len(derivative_keys) == 0 or order == 2
        ), "Second order forth derivatives supported"
        for key in derivative_keys:
            assert (
                len(key.derivatives) == 4
            ), f"Key with {len(key.derivatives)} deriv keys supplied to forth deriv"
            assert (
                key.derivatives[0]
                == key.derivatives[1]
                == key.derivatives[2]
                == key.derivatives[3]
            ), f"Mixed forth derivatives not supported"

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()
        for key in self.derivative_keys:
            indep_vars = key.derivatives
            if order == 2:
                self._stencil = self._stencil.union(
                    {
                        f"{indep_vars[0]}::-2",
                        f"{indep_vars[0]}::-1",
                        f"{indep_vars[0]}::0",
                        f"{indep_vars[0]}::1",
                        f"{indep_vars[0]}::2",
                    }
                )
                eval_list.append(ForthDerivO2(key))

        self._eval = torch.nn.ModuleList(eval_list)
