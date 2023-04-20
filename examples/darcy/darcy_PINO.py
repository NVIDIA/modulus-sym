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

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.key import Key
from modulus.sym.models.layers.spectral_layers import fourier_derivatives
from modulus.sym.node import Node

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.domain.validator import GridValidator
from modulus.sym.dataset import DictGridDataset
from modulus.sym.utils.io.plotter import GridValidatorPlotter
from modulus.sym.utils.io.vtk import grid_to_vtk

from utilities import download_FNO_dataset, load_FNO_dataset
from ops import dx, ddx


# [pde-loss]
class Darcy(torch.nn.Module):
    "Custom Darcy PDE definition for PINO"

    def __init__(self, gradient_method: str = "exact"):
        super().__init__()
        self.gradient_method = str(gradient_method)

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # get inputs
        u = input_var["sol"]
        c = input_var["coeff"]
        dcdx = input_var["Kcoeff_y"]  # data is reversed
        dcdy = input_var["Kcoeff_x"]

        dxf = 1.0 / u.shape[-2]
        dyf = 1.0 / u.shape[-1]
        # Compute gradients based on method
        # Exact first order and FDM second order
        if self.gradient_method == "exact":
            dudx_exact = input_var["sol__x"]
            dudy_exact = input_var["sol__y"]
            dduddx_exact = input_var["sol__x__x"]
            dduddy_exact = input_var["sol__y__y"]
            # compute darcy equation
            darcy = (
                1.0
                + (dcdx * dudx_exact)
                + (c * dduddx_exact)
                + (dcdy * dudy_exact)
                + (c * dduddy_exact)
            )
        # FDM gradients
        elif self.gradient_method == "fdm":
            dudx_fdm = dx(u, dx=dxf, channel=0, dim=0, order=1, padding="replication")
            dudy_fdm = dx(u, dx=dyf, channel=0, dim=1, order=1, padding="replication")
            dduddx_fdm = ddx(
                u, dx=dxf, channel=0, dim=0, order=1, padding="replication"
            )
            dduddy_fdm = ddx(
                u, dx=dyf, channel=0, dim=1, order=1, padding="replication"
            )
            # compute darcy equation
            darcy = (
                1.0
                + (dcdx * dudx_fdm)
                + (c * dduddx_fdm)
                + (dcdy * dudy_fdm)
                + (c * dduddy_fdm)
            )
        # Fourier derivative
        elif self.gradient_method == "fourier":
            dim_u_x = u.shape[2]
            dim_u_y = u.shape[3]
            u = F.pad(
                u, (0, dim_u_y - 1, 0, dim_u_x - 1), mode="reflect"
            )  # Constant seems to give best results
            f_du, f_ddu = fourier_derivatives(u, [2.0, 2.0])
            dudx_fourier = f_du[:, 0:1, :dim_u_x, :dim_u_y]
            dudy_fourier = f_du[:, 1:2, :dim_u_x, :dim_u_y]
            dduddx_fourier = f_ddu[:, 0:1, :dim_u_x, :dim_u_y]
            dduddy_fourier = f_ddu[:, 1:2, :dim_u_x, :dim_u_y]
            # compute darcy equation
            darcy = (
                1.0
                + (dcdx * dudx_fourier)
                + (c * dduddx_fourier)
                + (dcdy * dudy_fourier)
                + (c * dduddy_fourier)
            )
        else:
            raise ValueError(f"Derivative method {self.gradient_method} not supported.")

        # Zero outer boundary
        darcy = F.pad(darcy[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
        # Return darcy
        output_var = {
            "darcy": dxf * darcy,
        }  # weight boundary loss higher
        return output_var


# [pde-loss]


@modulus.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: ModulusConfig) -> None:
    # [datasets]
    # load training/ test data
    input_keys = [
        Key("coeff", scale=(7.48360e00, 4.49996e00)),
        Key("Kcoeff_x"),
        Key("Kcoeff_y"),
    ]
    output_keys = [
        Key("sol", scale=(5.74634e-03, 3.88433e-03)),
    ]

    download_FNO_dataset("Darcy_241", outdir="datasets/")
    invar_train, outvar_train = load_FNO_dataset(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth1.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=cfg.custom.ntrain,
    )
    invar_test, outvar_test = load_FNO_dataset(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth2.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=cfg.custom.ntest,
    )

    # add additional constraining values for darcy variable
    outvar_train["darcy"] = np.zeros_like(outvar_train["sol"])

    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)
    # [datasets]

    # [init-model]
    # Define FNO model
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys,
    )
    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=[input_keys[0]],
        decoder_net=decoder_net,
    )
    if cfg.custom.gradient_method == "exact":
        derivatives = [
            Key("sol", derivatives=[Key("x")]),
            Key("sol", derivatives=[Key("y")]),
            Key("sol", derivatives=[Key("x"), Key("x")]),
            Key("sol", derivatives=[Key("y"), Key("y")]),
        ]
        fno.add_pino_gradients(
            derivatives=derivatives,
            domain_length=[1.0, 1.0],
        )
    # [init-model]

    # [init-node]
    # Make custom Darcy residual node for PINO
    inputs = [
        "sol",
        "coeff",
        "Kcoeff_x",
        "Kcoeff_y",
    ]
    if cfg.custom.gradient_method == "exact":
        inputs += [
            "sol__x",
            "sol__y",
        ]
    darcy_node = Node(
        inputs=inputs,
        outputs=["darcy"],
        evaluate=Darcy(gradient_method=cfg.custom.gradient_method),
        name="Darcy Node",
    )
    nodes = [fno.make_node("fno"), darcy_node]
    # [init-node]

    # [constraint]
    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")
    # [constraint]

    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        requires_grad=True,
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
