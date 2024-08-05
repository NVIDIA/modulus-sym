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


from typing import Union, Dict, List, Callable

import numpy as np
import sympy as sp
from sympy import Symbol
import torch

from modulus.sym.dataset import ListIntegralDataset, ContinuousIntegralIterableDataset
from modulus.sym.utils.sympy import np_lambdify
from modulus.sym import Node
from modulus.sym.loss.loss import IntegralLossNorm, Loss
from modulus.sym.utils.io.plotter import _Plotter
from modulus.sym.domain.constraint.continuous import IntegralConstraint

from modulus.sym.geometry import Geometry, Parameterization


class SingleValue(torch.nn.Module):
    def __init__(self, value, name):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)), requires_grad=True)
        self.name = name

    def forward(self, *args, **kwargs):
        arg0 = args[0]
        keys = arg0.keys()
        dummy_key = list(keys)[0]
        dummy_val = arg0[dummy_key]
        return {self.name: self.value * torch.ones_like(dummy_val)}

    def get_device(self):
        return next(self.parameters()).device


class InferencerPlotterCustom(_Plotter):
    """
    Default plotter class for inferencer

    Bugfix for 1D
    """

    def __call__(self, invar, outvar):
        """Default function for plotting inferencer data"""
        from matplotlib import pyplot as plt

        ndim = len(invar)
        if ndim > 2:
            print("Default plotter can only handle <=2 input dimensions, passing")
            return []

        # interpolate 2D data onto grid
        if ndim == 2:
            extent, outvar = self._interpolate_2D(100, invar, outvar)

        # make plots
        dims = list(invar.keys())
        fs = []
        for k in outvar:
            f = plt.figure(figsize=(5, 4), dpi=100)
            if ndim == 1:
                plt.plot(invar[dims[0]][:, 0], outvar[k][:, 0])
                plt.xlabel(dims[0])
            elif ndim == 2:
                plt.imshow(outvar[k].T, origin="lower", extent=extent)
                plt.xlabel(dims[0])
                plt.ylabel(dims[1])
                plt.colorbar()
            plt.title(k)
            plt.tight_layout()
            fs.append((f, k))

        return fs


class IntegralInteriorConstraint(IntegralConstraint):
    """
    Integral Constraint applied to interior/volume of a geometry.

    TODO This could probably be made less redundant with the surface version

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.
    geometry : Geometry
        Modulus `Geometry` to apply the constraint with.
    outvar : Dict[str, Union[int, float, sp.Basic]]
        A dictionary of SymPy Symbols/Expr, floats or int.
        This is used to describe the constraint. For example,
        `outvar={'u': 0}` would specify the integral of `'u'`
        to be zero.
    batch_size : int
        Number of integrals to apply.
    integral_batch_size :  int
        Batch sized used in the Monte Carlo integration to compute
        the integral.
    criteria : Union[sp.basic, True]
        SymPy criteria function specifies to only integrate areas
        that satisfy this criteria. For example, if
        `criteria=sympy.Symbol('x')>0` then only areas that have positive
        `'x'` values will be integrated.
    lambda_weighting :  Dict[str, Union[int, float, sp.Basic]] = None
        The weighting of the constraint. For example,
        `lambda_weighting={'lambda_u': 2.0}` would
        weight the integral constraint by `2.0`.
    parameterization : Union[Parameterization, None]
        This allows adding parameterization or additional inputs.
    fixed_dataset : bool = True
        If True then the points sampled for this constraint are done right
        when initialized and fixed. If false then the points are continually
        resampled.
    batch_per_epoch : int = 100
        If `fixed_dataset=True` then the total number of integrals generated
        to apply constraint on is `total_nr_integrals=batch_per_epoch*batch_size`.
    quasirandom : bool = False
        If true then sample the points using the Halton sequence.
    num_workers : int
        Number of worker used in fetching data.
    loss : Loss
        Modulus `Loss` module that defines the loss type, (e.g. L2, L1, ...).
    shuffle : bool, optional
        Randomly shuffle examples in dataset every epoch, by default True
    """

    def __init__(
        self,
        nodes: List[Node],
        geometry: Geometry,
        outvar: Dict[str, Union[int, float, sp.Basic]],
        batch_size: int,
        integral_batch_size: int,
        criteria: Union[sp.Basic, Callable, None] = None,
        lambda_weighting: Dict[str, Union[int, float, sp.Basic]] = None,
        parameterization: Union[Parameterization, None] = None,
        fixed_dataset: bool = True,
        batch_per_epoch: int = 100,
        quasirandom: bool = False,
        num_workers: int = 0,
        loss: Loss = IntegralLossNorm(),
        shuffle: bool = True,
    ):

        # convert dict to parameterization if needed
        if parameterization is None:
            parameterization = geometry.parameterization
        elif isinstance(parameterization, dict):
            parameterization = Parameterization(parameterization)

        # Fixed number of integral examples
        if fixed_dataset:
            # sample geometry to generate integral batchs
            list_invar = []
            list_outvar = []
            list_lambda_weighting = []
            for i in range(batch_size * batch_per_epoch):
                # sample parameter ranges
                if parameterization:
                    specific_param_ranges = parameterization.sample(1)
                else:
                    specific_param_ranges = {}

                # sample boundary
                invar = geometry.sample_interior(
                    integral_batch_size,
                    criteria=criteria,
                    parameterization=Parameterization(
                        {
                            sp.Symbol(key): float(value)
                            for key, value in specific_param_ranges.items()
                        }
                    ),
                    quasirandom=quasirandom,
                )

                # compute outvar
                if (
                    not specific_param_ranges
                ):  # TODO this can be removed after a np_lambdify rewrite
                    specific_param_ranges = {"_": next(iter(invar.values()))[0:1]}
                outvar_star = _compute_outvar(specific_param_ranges, outvar)

                # set lambda weighting
                lambda_weighting_star = _compute_lambda_weighting(
                    specific_param_ranges, outvar, lambda_weighting
                )

                # store samples
                list_invar.append(invar)
                list_outvar.append(outvar_star)
                list_lambda_weighting.append(lambda_weighting_star)

            # make dataset of integral planes
            dataset = ListIntegralDataset(
                list_invar=list_invar,
                list_outvar=list_outvar,
                list_lambda_weighting=list_lambda_weighting,
            )
        # Continuous sampling
        else:
            # sample parameter ranges
            if parameterization:
                param_ranges_fn = lambda: parameterization.sample(1)
            else:
                param_ranges_fn = lambda: {}

            # invar function
            invar_fn = lambda param_range: geometry.sample_interior(
                integral_batch_size,
                criteria=criteria,
                parameterization=Parameterization(
                    {Symbol(key): float(value) for key, value in param_range.items()}
                ),
                quasirandom=quasirandom,
            )

            # outvar function
            outvar_fn = lambda param_range: _compute_outvar(param_range, outvar)

            # lambda weighting function
            lambda_weighting_fn = lambda param_range, outvar: _compute_lambda_weighting(
                param_range, outvar, lambda_weighting
            )

            # make dataset of integral planes
            dataset = ContinuousIntegralIterableDataset(
                invar_fn=invar_fn,
                outvar_fn=outvar_fn,
                batch_size=batch_size,
                lambda_weighting_fn=lambda_weighting_fn,
                param_ranges_fn=param_ranges_fn,
            )

        self.batch_size = batch_size

        # initialize constraint
        super().__init__(
            nodes=nodes,
            dataset=dataset,
            loss=loss,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
        )


class ProductModule(torch.nn.Module):
    """
    Simple module that multiplies an input by a known function of the inputs.

    Intended for evaluating the inner product of a function being learned
    with a known function, by using IntegralInteriorConstraint.
    """

    def __init__(self, input_key, output_key, func):
        super().__init__()
        self._input_key = input_key
        self._output_key = output_key
        self.func = func

    def forward(self, vars):
        output_val = vars[self._input_key] * self.func(vars)
        return {self._output_key: output_val}


def build_orthogonal_function_nodes(input_keys, output_key, mode_ns, box_width):
    """
    Build a list of nodes which calculate the product of the `output_key` (eg `psi`)
    with input functions. Here we use sin functions as the input functions.
    """
    orth_func_nodes = []
    norm_factor = np.sqrt(box_width * np.pi / 2)
    for mode_n in mode_ns:
        orth_func = (
            lambda vars: torch.sin(mode_n * np.pi * vars["x"] / box_width) / norm_factor
        )
        mode_output_key = f"func_product_{mode_n:02d}"
        orth_module = ProductModule(output_key, mode_output_key, orth_func)
        orth_func_node = Node(
            inputs=input_keys + [output_key],
            outputs=[mode_output_key],
            evaluate=orth_module,
            name=mode_output_key,
            optimize=False,
        )
        orth_func_nodes.append(orth_func_node)
    return orth_func_nodes


def _compute_outvar(invar, outvar_sympy):
    outvar = {}
    for key in outvar_sympy.keys():
        outvar[key] = np_lambdify(outvar_sympy[key], {**invar})(**invar)
    return outvar


def _compute_lambda_weighting(invar, outvar, lambda_weighting_sympy):
    lambda_weighting = {}
    if lambda_weighting_sympy is None:
        for key in outvar.keys():
            lambda_weighting[key] = np.ones_like(next(iter(invar.values())))
    else:
        for key in outvar.keys():
            lambda_weighting[key] = np_lambdify(
                lambda_weighting_sympy[key], {**invar, **outvar}
            )(**invar, **outvar)
    return lambda_weighting
