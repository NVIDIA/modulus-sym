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
"""
@Author : Clement Etienam
"""
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
import os
import modulus
from modulus.sym.hydra import ModulusConfig
from modulus.sym.hydra import to_absolute_path
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.domain.validator import GridValidator
from modulus.sym.dataset import DictGridDataset
from modulus.sym.utils.io.plotter import GridValidatorPlotter
from NVRS import *
from utilities import load_FNO_dataset2, preprocess_FNO_mat
from ops import dx, ddx
from modulus.sym.models.afno.afno import *
import shutil
import cupy as cp
from sklearn.model_selection import train_test_split
import scipy.io as sio
import requests
from modulus.sym.utils.io.plotter import ValidatorPlotter

torch.set_default_dtype(torch.float32)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, "confirm": 1}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


class CustomValidatorPlotterP(ValidatorPlotter):
    def __init__(self, timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt):
        self.timmee = timmee
        self.max_t = max_t
        self.MAXZ = MAXZ
        self.pini_alt = pini_alt
        self.nx = nx
        self.ny = ny
        self.wells = wells
        self.steppi = steppi
        self.tc2 = tc2
        self.dt = dt

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables

        pressure_true, pressure_pred = true_outvar["pressure"], pred_outvar["pressure"]

        # make plot
        f_big = []
        for itt in range(self.steppi):
            look = (pressure_pred[0, itt, :, :]) * self.pini_alt

            lookf = (pressure_true[0, itt, :, :]) * self.pini_alt

            diff1 = abs(look - lookf)

            XX, YY = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
            f_2 = plt.figure(figsize=(12, 12), dpi=100)
            plt.subplot(2, 2, 1)
            plt.pcolormesh(XX.T, YY.T, look, cmap="jet")
            plt.title("Pressure AFNO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(np.min(np.reshape(lookf, (-1,))), np.max(np.reshape(lookf, (-1,))))
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(2, 2, 2)
            plt.pcolormesh(XX.T, YY.T, lookf, cmap="jet")
            plt.title("Pressure CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(2, 2, 3)
            plt.pcolormesh(XX.T, YY.T, diff1, cmap="jet")
            plt.title("Pressure (CFD - AFNO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = "Timestep --" + str(int((itt + 1) * self.dt * self.MAXZ)) + " days"

            plt.suptitle(tita, fontsize=16)

            # name = namet + str(int(itt)) + '.png'
            # plt.savefig(name)
            # #plt.show()
            # plt.clf()
            namez = "pressure_simulations" + str(int(itt))
            yes = (f_2, namez)
            f_big.append(yes)
        return f_big


class CustomValidatorPlotterS(ValidatorPlotter):
    def __init__(self, timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt):
        self.timmee = timmee
        self.max_t = max_t
        self.MAXZ = MAXZ
        self.pini_alt = pini_alt
        self.nx = nx
        self.ny = ny
        self.wells = wells
        self.steppi = steppi
        self.tc2 = tc2
        self.dt = dt

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables

        water_true, water_pred = true_outvar["water_sat"], pred_outvar["water_sat"]

        # make plot

        f_big = []
        for itt in range(self.steppi):

            XX, YY = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
            f_2 = plt.figure(figsize=(12, 12), dpi=100)

            look_sat = water_pred[0, itt, :, :]
            look_oil = 1 - look_sat

            lookf_sat = water_true[0, itt, :, :]
            lookf_oil = 1 - lookf_sat

            diff1_wat = abs(look_sat - lookf_sat)
            diff1_oil = abs(look_oil - lookf_oil)

            plt.subplot(2, 3, 1)
            plt.pcolormesh(XX.T, YY.T, look_sat, cmap="jet")
            plt.title("water_sat AFNO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf_sat, (-1,))),
                np.max(np.reshape(lookf_sat, (-1,))),
            )
            cbar1.ax.set_ylabel(" water_sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(2, 3, 2)
            plt.pcolormesh(XX.T, YY.T, lookf_sat, cmap="jet")
            plt.title("water_sat CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" water sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(2, 3, 3)
            plt.pcolormesh(XX.T, YY.T, diff1_wat, cmap="jet")
            plt.title("water_sat (CFD - AFNO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" water sat ", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(2, 3, 4)
            plt.pcolormesh(XX.T, YY.T, look_oil, cmap="jet")
            plt.title("oil_sat AFNO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf_oil, (-1,))),
                np.max(np.reshape(lookf_oil, (-1,))),
            )
            cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(2, 3, 5)
            plt.pcolormesh(XX.T, YY.T, lookf_oil, cmap="jet")
            plt.title("oil_sat CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" oil sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(2, 3, 6)
            plt.pcolormesh(XX.T, YY.T, diff1_oil, cmap="jet")
            plt.title("oil_sat (CFD - AFNO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" oil sat ", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = "Timestep --" + str(int((itt + 1) * self.dt * self.MAXZ)) + " days"

            plt.suptitle(tita, fontsize=16)

            # name = namet + str(int(itt)) + '.png'
            # plt.savefig(name)
            # #plt.show()
            # plt.clf()
            namez = "saturation_simulations" + str(int(itt))
            yes = (f_2, namez)
            f_big.append(yes)
        return f_big


# [pde-loss]
# define custom class for black oil model
class Black_oil(torch.nn.Module):
    "Custom Black oil PDE definition for AFNO"

    def __init__(
        self,
        UIR,
        pini_alt,
        LUB,
        HUB,
        aay,
        bby,
        SWI,
        SWR,
        UW,
        BW,
        UO,
        BO,
        MAXZ,
        nx,
        ny,
        approach,
    ):
        super().__init__()
        self.UIR = UIR
        self.UWR = UIR
        self.pini_alt = pini_alt
        self.LUB = LUB
        self.HUB = HUB
        self.aay = aay
        self.bby = bby
        self.SWI = SWI
        self.SWR = SWR
        self.UW = UW
        self.BW = BW
        self.UO = UO
        self.BO = BO
        self.MAXZ = MAXZ
        self.nx = nx
        self.ny = ny
        self.approach = approach

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # get inputs

        u = input_var["pressure"]
        perm = input_var["perm"]
        fin = input_var["Q"]
        finwater = input_var["Qw"]
        dt = input_var["Time"]
        pini = input_var["Pini"]
        poro = input_var["Phi"]
        sini = input_var["Swini"]
        sat = input_var["water_sat"]

        if self.approach == 3:
            dudx_exact = input_var["pressure__x"] * self.pini_alt
            dudy_exact = input_var["pressure__y"] * self.pini_alt
            dduddx_exact = input_var["pressure__x__x"] * self.pini_alt
            dduddy_exact = input_var["pressure__y__y"] * self.pini_alt

        siniuse = sini[0, 0, 0, 0]

        dtin = dt * self.MAXZ
        dxf = 1.0 / u.shape[3]

        if self.approach == 1:

            u = u * self.pini_alt
            pini = pini * self.pini_alt
            # Pressure equation Loss
            fin = fin * self.UIR
            finwater = finwater * self.UIR
            cuda = 0
            device = torch.device(
                f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
            )

            # print(pressurey.shape)
            p_loss = torch.zeros_like(u).to(device, torch.float32)
            s_loss = torch.zeros_like(u).to(device, torch.float32)

            a = perm  # absolute permeability
            v_min, v_max = self.LUB, self.HUB
            new_min, new_max = self.aay, self.bby

            m = (new_max - new_min) / (v_max - v_min)
            b = new_min - m * v_min
            a = m * a + b

            finuse = fin
            finusew = finwater
            dta = dtin

            pressure = u
            # water_sat = sat

            prior_pressure = torch.zeros(
                sat.shape[0], sat.shape[1], self.nx, self.ny
            ).to(device, torch.float32)
            prior_pressure[:, 0, :, :] = self.pini_alt * (
                torch.ones(sat.shape[0], self.nx, self.ny).to(device, torch.float32)
            )
            prior_pressure[:, 1:, :, :] = u[:, :-1, :, :]

            # dsp = u - prior_pressure  #dp

            prior_sat = torch.zeros(sat.shape[0], sat.shape[1], self.nx, self.ny).to(
                device, torch.float32
            )
            prior_sat[:, 0, :, :] = siniuse * (
                torch.ones(sat.shape[0], self.nx, self.ny).to(device, torch.float32)
            )
            prior_sat[:, 1:, :, :] = sat[:, :-1, :, :]

            dsw = sat - prior_sat  # ds
            dsw = torch.clip(dsw, 0.001, None)

            S = torch.div(
                torch.sub(prior_sat, self.SWI, alpha=1), (1 - self.SWI - self.SWR)
            )

            # Pressure equation Loss
            Mw = torch.divide(torch.square(S), (self.UW * self.BW))  # Water mobility
            Mo = torch.div(
                torch.square(torch.sub(torch.ones(S.shape, device=u.device), S)),
                (self.UO * self.BO),
            )

            krw = torch.square(S)
            kroil = torch.square(torch.sub(torch.ones(S.shape, device=u.device), S))
            Mt = Mw + Mo
            a1 = torch.mul(Mt, a)  # overall Effective permeability
            a1water = torch.mul(Mw, a)  # water Effective permeability

            # compute first dffrential
            gulpa = []
            gulp2a = []
            for m in range(sat.shape[0]):  # Batch
                inn_now = pressure[m, :, :, :][:, None, :, :]
                dudx_fdma = dx(
                    inn_now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = dx(
                    inn_now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulpa.append(dudx_fdma)
                gulp2a.append(dudy_fdma)
            dudx_fdm = torch.stack(gulpa, 0)[:, :, 0, :, :]
            dudy_fdm = torch.stack(gulp2a, 0)[:, :, 0, :, :]

            # Compute second diffrential

            gulpa = []
            gulp2a = []
            for m in range(sat.shape[0]):  # Batch
                inn_now = pressure[m, :, :, :][:, None, :, :]
                dudx_fdma = ddx(
                    inn_now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = ddx(
                    inn_now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulpa.append(dudx_fdma)
                gulp2a.append(dudy_fdma)
            dduddx_fdm = torch.stack(gulpa, 0)[:, :, 0, :, :]
            dduddy_fdm = torch.stack(gulp2a, 0)[:, :, 0, :, :]

            inn_now2 = a1
            dcdx = dx(
                inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
            )
            dcdy = dx(
                inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
            )

            darcy_pressure = (
                fin
                + (dcdx * dudx_fdm)
                + (a1 * dduddx_fdm)
                + (dcdy * dudy_fdm)
                + (a1 * dduddy_fdm)
            )

            # Zero outer boundary
            # darcy_pressure = F.pad(darcy_pressure[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
            darcy_pressure = dxf * darcy_pressure * 1e-7

            p_loss = darcy_pressure

            # Saruration equation loss
            dudx = dudx_fdm
            dudy = dudy_fdm

            dduddx = dduddx_fdm
            dduddy = dduddy_fdm

            inn_now2 = a1water
            dadx = dx(
                inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
            )
            dady = dx(
                inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
            )

            flux = (
                (dadx * dudx) + (a1water * dduddx) + (dady * dudy) + (a1water * dduddy)
            )
            fifth = poro * (dsw / dta)
            toge = flux + finusew
            darcy_saturation = fifth - toge

            # Zero outer boundary
            # darcy_saturation = F.pad(darcy_saturation[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
            darcy_saturation = dxf * darcy_saturation * 1e-7

            s_loss = darcy_saturation

        # Slower but more accurate implementation
        elif self.approach == 2:
            u = u * self.pini_alt
            pini = pini * self.pini_alt
            # Pressure equation Loss
            fin = fin * self.UIR
            finwater = finwater * self.UIR
            cuda = 0
            device = torch.device(
                f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
            )

            # print(pressurey.shape)
            p_loss = torch.zeros_like(u).to(device, torch.float32)
            s_loss = torch.zeros_like(u).to(device, torch.float32)
            # print(sat.shape)
            # output_var  = dict()
            for zig in range(sat.shape[0]):
                for count in range(sat.shape[1]):
                    if count == 0:
                        prior_sat = sini[zig, 0, :, :][None, None, :, :]
                        prior_pressure = pini[zig, 0, :, :][None, None, :, :]
                    else:
                        prior_sat = sat[zig, (count - 1), :, :][None, None, :, :]
                        prior_pressure = u[zig, count - 1, :, :][None, None, :, :]

                    pressure = u[zig, count, :, :][None, None, :, :]
                    water_sat = sat[zig, count, :, :][None, None, :, :]

                    finuse = fin
                    a = perm[zig, 0, :, :][None, None, :, :]
                    v_min, v_max = self.LUB, self.HUB
                    new_min, new_max = self.aay, self.bby

                    m = (new_max - new_min) / (v_max - v_min)
                    b = new_min - m * v_min
                    a = m * a + b
                    S = torch.div(
                        torch.sub(prior_sat, self.SWI, alpha=1),
                        (1 - self.SWI - self.SWR),
                    )

                    # Pressure equation Loss

                    Mw = torch.divide(
                        torch.square(S), (self.UW * self.BW)
                    )  # Water mobility
                    Mo = torch.div(
                        torch.square(
                            torch.sub(torch.ones(S.shape, device=u.device), S)
                        ),
                        (self.UO * self.BO),
                    )
                    Mt = Mw + Mo
                    a1 = torch.mul(Mt, a)  # Effective permeability

                    ua = pressure
                    a2 = a1

                    dyf = 1.0 / u.shape[3]

                    # FDM gradients
                    dudx_fdm = dx(
                        ua, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                    )
                    dudy_fdm = dx(
                        ua, dx=dyf, channel=0, dim=1, order=1, padding="replication"
                    )
                    dduddx_fdm = ddx(
                        ua, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                    )
                    dduddy_fdm = ddx(
                        ua, dx=dyf, channel=0, dim=1, order=1, padding="replication"
                    )

                    dcdx = dx(
                        a2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                    )
                    dcdy = dx(
                        a2, dx=dyf, channel=0, dim=1, order=1, padding="replication"
                    )

                    # compute darcy equation
                    darcy_pressure = (
                        finuse[zig, 0, :, :][None, None, :, :]
                        + (dcdx * dudx_fdm)
                        + (a2 * dduddx_fdm)
                        + (dcdy * dudy_fdm)
                        + (a2 * dduddy_fdm)
                    )

                    # Zero outer boundary
                    # darcy_pressure = F.pad(darcy_pressure[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
                    darcy_pressure = dxf * darcy_pressure * 1e-7

                    p_loss[zig, count, :, :] = darcy_pressure

                    # output_var["darcy_pressure"] = torch.mean(p_loss,dim = 0)[None,:,:,:]

                    # Saturation equation Loss

                    finuse = finwater[zig, 0, :, :][None, None, :, :]
                    dsw = water_sat - prior_sat
                    dsw = torch.clip(dsw, 0.001, None)

                    dta = dtin[zig, 0, :, :][None, None, :, :]

                    Mw = torch.divide(
                        torch.square(S), (self.UW * self.BW)
                    )  # Water mobility
                    Mt = Mw
                    a1 = torch.mul(Mt, a)  # Effective permeability to water

                    ua = pressure
                    a2 = a1

                    dudx = dx(
                        ua, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                    )
                    dudy = dx(
                        ua, dx=dyf, channel=0, dim=1, order=1, padding="replication"
                    )

                    dduddx = ddx(
                        ua, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                    )
                    dduddy = ddx(
                        ua, dx=dyf, channel=0, dim=1, order=1, padding="replication"
                    )

                    dadx = dx(
                        a2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                    )
                    dady = dx(
                        a2, dx=dyf, channel=0, dim=1, order=1, padding="replication"
                    )

                    flux = (dadx * dudx) + (a2 * dduddx) + (dady * dudy) + (a2 * dduddy)
                    # flux = flux[:,0,:,:]
                    # temp = dsw_dt
                    # fourth = poro * CFW * prior_sat * (dsp/dta)
                    fifth = poro[zig, 0, :, :][None, None, :, :] * (dsw / dta)
                    toge = flux + finuse
                    darcy_saturation = fifth - toge

                    # Zero outer boundary
                    # darcy_saturation = F.pad(darcy_saturation[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
                    darcy_saturation = dxf * darcy_saturation * 1e-7

                    # print(darcy_saturation.shape)

                    s_loss[zig, count, :, :] = darcy_saturation

        else:

            u = u * self.pini_alt
            pini = pini * self.pini_alt
            # Pressure equation Loss
            fin = fin * self.UIR
            finwater = finwater * self.UIR
            cuda = 0
            device = torch.device(
                f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
            )

            # print(pressurey.shape)
            p_loss = torch.zeros_like(u).to(device, torch.float32)
            s_loss = torch.zeros_like(u).to(device, torch.float32)

            a = perm  # absolute permeability
            v_min, v_max = self.LUB, self.HUB
            new_min, new_max = self.aay, self.bby

            m = (new_max - new_min) / (v_max - v_min)
            b = new_min - m * v_min
            a = m * a + b

            finuse = fin
            finusew = finwater
            dta = dtin

            pressure = u
            # water_sat = sat

            prior_pressure = torch.zeros(
                sat.shape[0], sat.shape[1], self.nx, self.ny
            ).to(device, torch.float32)
            prior_pressure[:, 0, :, :] = self.pini_alt * (
                torch.ones(sat.shape[0], self.nx, self.ny).to(device, torch.float32)
            )
            prior_pressure[:, 1:, :, :] = u[:, :-1, :, :]

            # dsp = u - prior_pressure  #dp

            prior_sat = torch.zeros(sat.shape[0], sat.shape[1], self.nx, self.ny).to(
                device, torch.float32
            )
            prior_sat[:, 0, :, :] = siniuse * (
                torch.ones(sat.shape[0], self.nx, self.ny).to(device, torch.float32)
            )
            prior_sat[:, 1:, :, :] = sat[:, :-1, :, :]

            dsw = sat - prior_sat  # ds
            dsw = torch.clip(dsw, 0.001, None)

            S = torch.div(
                torch.sub(prior_sat, self.SWI, alpha=1), (1 - self.SWI - self.SWR)
            )

            # Pressure equation Loss
            Mw = torch.divide(torch.square(S), (self.UW * self.BW))  # Water mobility
            Mo = torch.div(
                torch.square(torch.sub(torch.ones(S.shape, device=u.device), S)),
                (self.UO * self.BO),
            )

            krw = torch.square(S)
            kroil = torch.square(torch.sub(torch.ones(S.shape, device=u.device), S))
            Mt = Mw + Mo
            a1 = torch.mul(Mt, a)  # overall Effective permeability
            a1water = torch.mul(Mw, a)  # water Effective permeability

            # compute first dffrential

            dudx_fdm = dudx_exact
            dudy_fdm = dudy_exact

            # Compute second diffrential

            dduddx_fdm = dduddx_exact
            dduddy_fdm = dduddy_exact

            inn_now2 = a1
            dcdx = ddx(
                inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
            )
            dcdy = ddx(
                inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
            )

            darcy_pressure = (
                fin
                + (dcdx * dudx_fdm)
                + (a1 * dduddx_fdm)
                + (dcdy * dudy_fdm)
                + (a1 * dduddy_fdm)
            )

            # Zero outer boundary
            # darcy_pressure = F.pad(darcy_pressure[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
            darcy_pressure = dxf * darcy_pressure * 1e-7

            p_loss = darcy_pressure

            # Saruration equation loss
            dudx = dudx_fdm
            dudy = dudy_fdm

            dduddx = dduddx_fdm
            dduddy = dduddy_fdm

            inn_now2 = a1water
            dadx = ddx(
                inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
            )
            dady = ddx(
                inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
            )

            flux = (
                (dadx * dudx) + (a1water * dduddx) + (dady * dudy) + (a1water * dduddy)
            )
            fifth = poro * (dsw / dta)
            toge = flux + finusew
            darcy_saturation = fifth - toge

            # Zero outer boundary
            # darcy_saturation = F.pad(darcy_saturation[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
            darcy_saturation = dxf * darcy_saturation * 1e-7

            s_loss = darcy_saturation

        output_var = {"pressured": p_loss, "saturationd": s_loss}
        return output_var


# [pde-loss]
@modulus.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: ModulusConfig) -> None:
    print("")
    print("------------------------------------------------------------------")
    print("")
    print("\n")
    print("|-----------------------------------------------------------------|")
    print("|   TRAIN THE MODEL USING A 2D PHYSICS DRIVEN AFNO APPROACH:      |")
    print("|-----------------------------------------------------------------|")
    print("")
    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    default = None
    while True:
        default = int(
            input("Select 1 = use default values | 2 = Use user defined values \n")
        )
        if (default > 2) or (default < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:

            break

    if not os.path.exists(to_absolute_path("../PACKETS")):
        os.makedirs(to_absolute_path("../PACKETS"))
    else:
        pass

    if default == 1:
        print(" --------------Use constrained presure residual method--------- ")
        method = 5
        typee = 0
    else:

        method = None
        while True:
            method = cp.int(
                input(
                    "Select inverse problem solution method for pressure & saturation \
            solve -\n\
            1 = GMRES\n\
            2 = spsolve\n\
            3 = Conjugate Gradient\n\
            4 = LSQR\n\
            5 = CPR\n\
            6 = AMGX\n: "
                )
            )
            if (method > 6) or (method < 1):
                # raise SyntaxError('please select value between 1-2')
                print("")
                print("please try again and select value between 1-6")
            else:

                break

        if method == 6:
            typee = cp.int(
                input(
                    "Select JSON config -\n\
        1 = Standard\n\
        2 = Exotic\n\
        3 = DILU\n\
        4 = Aggregation DS\n\
        5 = Aggregation Jacobi\n\
        6 = Aggregation Thrust\n\
        7 = Aggregation Conjugate gradient\n\
        8 = PBICGSTAB_Aggregation with JACOBI smmothing\n\
        9 = CG DILU\n\
        10 = GMRES AMG\n: "
                )
            )
        else:
            typee = 0

    if default == 1:
        approach = 1
    else:
        approach = None
        while True:
            print("Remark: Option 3 is not fully computed")
            approach = int(
                input(
                    "Select computation of spatial gradients -\n\
            1 = Approximate and fast computation\n\
            2 = Exact but slighly slower computation using FDM\n\
            3 = Exact gradient using FNO\n: "
                )
            )
            if (approach > 3) or (approach < 1):
                # raise SyntaxError('please select value between 1-2')
                print("")
                print("please try again and select value between 1-3")
            else:

                break

    # Varaibles needed for NVRS
    nx, ny, nz = 33, 33, 1
    BO = 1.1  # oil formation volume factor
    BW = 1  # Water formation volume factor
    UW = 1.0  # water viscosity in cP
    UO = 2.5  # oil viscosity in cP
    DX = 50.0  # size of pixel in x direction
    DY = 50.0  # sixze of pixel in y direction
    DZ = 20.0  # sizze of pixel in z direction

    DX = cp.float32(DX)
    DY = cp.float32(DY)
    UW = cp.float32(1)  # water viscosity in cP
    UO = cp.float32(2.5)  # oil viscosity in cP
    SWI = cp.float32(0.1)
    SWR = cp.float32(0.1)
    CFO = cp.float32(1e-5)  # oil compressibility in 1/psi
    IWSw = 0.2  # initial water saturation
    pini_alt = 1e3
    P1 = cp.float32(pini_alt)  # Bubble point pressure psia
    PB = P1
    mpor, hpor = 0.05, 0.5  # minimum and maximum porosity
    BW = cp.float32(BW)  # Water formation volume factor
    BO = cp.float32(BO)  # Oil formation volume factor
    PATM = cp.float32(14.6959)  # Atmospheric pressure in psi

    # training
    LUB, HUB = 1e-1, 1  # Permeability rescale
    aay, bby = 50, 500  # Permeability range mD
    Low_K, High_K = aay, bby

    batch_size = 1000  #'size of simulated labelled data to run'
    timmee = 100.0  # float(input ('Enter the time step interval duration for simulation (days): '))
    max_t = (
        3000.0  # float(input ('Enter the maximum time in days for simulation(days): '))
    )
    MAXZ = 6000  # reference maximum time in days of simulation
    steppi = int(max_t / timmee)
    choice = 1  #  1= Non-Gaussian prior, 2 = Gaussian prior
    factorr = 0.1  # from [0 1] excluding the limits for PermZ
    LIR = 200  # lower injection rate
    UIR = 2000  # uppwer injection rate
    input_channel = 7  # [Perm, Q,QW,Phi,dt, initial_pressure, initial_water_sat]

    # dt = timmee/max_t
    print("Check the condition  CFL number")
    ratioo = timmee / DX
    if ratioo <= 1:
        print("Solution is unconditionally stable with time step, use IMPES method")
        CFL = 1  # IMPES
    else:
        print(
            "Solution is conditionally stable with time step, Using Fully-Implicit method"
        )
        CFL = 2  # IMPLICT
        # step2 = int(input ('Enter the chop interval for time calculation (3-20): '))
        step2 = int(5)

    # tc2 = Equivalent_time(timmee,2100,timmee,max_t)
    tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
    dt = np.diff(tc2)[0]  # Time-step
    # 4 injector and 4 producer wells
    wells = np.array(
        [
            1,
            24,
            1,
            3,
            3,
            1,
            31,
            1,
            1,
            31,
            31,
            1,
            7,
            9,
            2,
            14,
            12,
            2,
            28,
            19,
            2,
            14,
            27,
            2,
        ]
    )
    wells = np.reshape(wells, (-1, 3), "C")

    # Get prior density of permeability and porosity field

    if not os.path.exists(to_absolute_path("../Training_Images")):
        os.makedirs(to_absolute_path("../Training_Images"))
    else:
        pass

    if default == 1:
        use_pretrained = 1
    else:
        use_pretrained = 2

    if use_pretrained == 1:
        print("Use already generated ensemble from Google drive folder")
        if choice == 1:
            bb = os.path.isfile(to_absolute_path("../PACKETS/Ganensemble.mat"))
            if bb == False:
                print("Get initial geology from saved Multiple-point-statistics run")

                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1w81M5M2S0PD9CF2761dxmiKQ5c0OFPaH",
                    to_absolute_path("../PACKETS/Ganensemble.mat"),
                )
                print("...Downlaod completed.......")

                filename = to_absolute_path(
                    "../PACKETS/Ganensemble.mat"
                )  # Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                scaler1a = MinMaxScaler(feature_range=(aay, bby))
                (scaler1a.fit(ini_ensemblef))
                ini_ensemblef = scaler1a.transform(ini_ensemblef)
            else:
                filename = to_absolute_path(
                    "../PACKETS/Ganensemble.mat"
                )  # Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                scaler1a = MinMaxScaler(feature_range=(aay, bby))
                (scaler1a.fit(ini_ensemblef))
                ini_ensemblef = scaler1a.transform(ini_ensemblef)
        else:
            bb = os.path.isfile(to_absolute_path("../PACKETS/Ganensemble_gauss.mat"))
            if bb == False:
                print("Get initial geology from saved Two - point-statistics run")
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1Kbe3F7XRzubmDU2bPkfHo2bEmjBNIZ29",
                    to_absolute_path("../PACKETS/Ganensemble_gauss.mat"),
                )
                print("...Downlaod completed.......")

                filename = to_absolute_path(
                    "../PACKETS/Ganensemble_gauss.mat"
                )  # Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                scaler1a = MinMaxScaler(feature_range=(aay, bby))
                (scaler1a.fit(ini_ensemblef))
                ini_ensemblef = scaler1a.transform(ini_ensemblef)
            else:
                filename = to_absolute_path(
                    "../PACKETS/Ganensemble_gauss.mat"
                )  # Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                scaler1a = MinMaxScaler(feature_range=(aay, bby))
                (scaler1a.fit(ini_ensemblef))
                ini_ensemblef = scaler1a.transform(ini_ensemblef)
    else:
        print("Generated prior ensemble from training images")
        if choice == 1:
            print("Get initial geology from scratch using Multiple-point-statistics")
            TII = 3  # Training image
            if TII == 3:
                print("TI = 3")

                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1bib2JAZfBpW4bKz5LdCmhAOtxQixkTzj",
                    to_absolute_path("../Training_Images/iglesias2.out"),
                )
                print("...Downlaod completed.......")

                kq = np.genfromtxt(
                    to_absolute_path("../Training_Images/iglesias2.out"),
                    skip_header=0,
                    dtype="float",
                )
                kq = kq.reshape(-1, 1)
                clfy = MinMaxScaler(feature_range=(Low_K, High_K))
                (clfy.fit(kq))
                kq = np.reshape(kq, (nx, ny), "F")
                kjennq = kq
                permx = kjennq

            else:

                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1TrAVvB-XXCzwHqDdCR4BJnmoe8nPsWIF",
                    to_absolute_path("../Training_Images/TI_2.out"),
                )
                print("...Downlaod completed.......")

                kq = np.genfromtxt(
                    to_absolute_path("../Training_Images/TI_2.out"),
                    skip_header=3,
                    dtype="float",
                )
                kq = kq.reshape(-1, 1)
                clfy = MinMaxScaler(feature_range=(Low_K, High_K))
                (clfy.fit(kq))
                kq = clfy.transform(kq)
                kq = np.reshape(kq, (250, 250), "F")
                kjennq = kq.T
                permx = kjennq

            N_tot = batch_size
            see = intial_ensemble(nx, ny, nz, N_tot, permx)

            ini_ensemblef = see
            sio.savemat(
                to_absolute_path("../PACKETS/Ganensemble.mat"), {"Z": ini_ensemblef}
            )
        else:
            print("Get initial geology from scratch using Two-point-statistics run")
            ini_ensemblef = initial_ensemble_gaussian(nx, ny, nz, batch_size, aay, bby)

            scaler1a = MinMaxScaler(feature_range=(aay, bby))
            (scaler1a.fit(ini_ensemblef))
            ini_ensemblef = scaler1a.transform(ini_ensemblef)
            sio.savemat(
                to_absolute_path("../PACKETS/Ganensemble_gauss.mat"),
                {"Z": ini_ensembleee},
            )

    X_train, X_test = train_test_split(
        ini_ensemblef.T, train_size=batch_size + 1, random_state=42
    )
    X_train = X_train.T
    X_test = X_test.T

    index = batch_size
    imp = batch_size

    bb = os.path.isfile(to_absolute_path("../PACKETS/Training4.mat"))
    if bb == False:
        if use_pretrained == 1:

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1I-27_S53ORRFB_hIN_41r3Ntc6PpOE40",
                to_absolute_path("../PACKETS/Training4.mat"),
            )
            print("...Downlaod completed.......")

            print("Load simulated labelled training data from MAT file")
            matt = sio.loadmat(to_absolute_path("../PACKETS/Training4.mat"))
            X_data1 = matt["INPUT"]
            data_use1 = matt["OUTPUT"]
        else:
            print("Get simulated training data from scratch")
            index = np.random.choice(X_train.shape[1], imp, replace=False)
            X_data1, data_use1 = Get_actual_few(
                X_train[:, index],
                nx,
                ny,
                nz,
                max_t,
                DX,
                DY,
                DZ,
                BO,
                BW,
                CFL,
                timmee,
                MAXZ,
                factorr,
                steppi,
                LIR,
                UIR,
                LUB,
                HUB,
                aay,
                bby,
                mpor,
                hpor,
                dt,
                IWSw,
                PB,
                PATM,
                CFO,
                method,
                SWI,
                SWR,
                UW,
                UO,
                typee,
                step2,
                pini_alt,
                input_channel,
                2,
            )

            sio.savemat(
                to_absolute_path("../PACKETS/Training4.mat"),
                {"INPUT": X_data1, "OUTPUT": data_use1},
            )
    else:
        print("Load simulated labelled training data from MAT file")
        matt = sio.loadmat(to_absolute_path("../PACKETS/Training4.mat"))
        X_data1 = matt["INPUT"]
        data_use1 = matt["OUTPUT"]

    cPerm = np.zeros((X_data1.shape[0], 1, nx, ny))  # Permeability
    cQ = np.zeros((X_data1.shape[0], 1, nx, ny))  # Overall source/sink term
    cQw = np.zeros((X_data1.shape[0], 1, nx, ny))  # Sink term
    cPhi = np.zeros((X_data1.shape[0], 1, nx, ny))  # Porosity
    cTime = np.zeros((X_data1.shape[0], 1, nx, ny))  # Time index
    cPini = np.zeros((X_data1.shape[0], 1, nx, ny))  # Initial pressure
    cSini = np.zeros((X_data1.shape[0], 1, nx, ny))  # Initial water saturation

    cPress = np.zeros((X_data1.shape[0], steppi, nx, ny))  # Pressure
    cSat = np.zeros((X_data1.shape[0], steppi, nx, ny))  # Water saturation

    for kk in range(X_data1.shape[0]):
        perm = X_data1[kk, 0, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPerm[kk, :, :, :] = permin

        perm = X_data1[kk, 1, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQ[kk, :, :, :] = permin

        perm = X_data1[kk, 2, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQw[kk, :, :, :] = permin

        perm = X_data1[kk, 3, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPhi[kk, :, :, :] = permin

        perm = X_data1[kk, 4, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cTime[kk, :, :, :] = permin

        perm = X_data1[kk, 5, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPini[kk, :, :, :] = permin

        perm = X_data1[kk, 6, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cSini[kk, :, :, :] = permin

        perm = data_use1[kk, :steppi, :, :]
        cPress[kk, :, :, :] = np.clip(perm, 1 / pini_alt, 1.0)

        perm = data_use1[kk, steppi:, :, :]
        cSat[kk, :, :, :] = perm

    sio.savemat(
        to_absolute_path("../PACKETS/simulations.mat"),
        {
            "perm": cPerm,
            "Q": cQ,
            "Qw": cQw,
            "Phi": cPhi,
            "Time": cTime,
            "Pini": cPini,
            "Swini": cSini,
            "pressure": cPress,
            "water_sat": cSat,
        },
    )

    preprocess_FNO_mat(to_absolute_path("../PACKETS/simulations.mat"))

    bb = os.path.isfile(to_absolute_path("../PACKETS/iglesias2.out"))
    if bb == False:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1_9VRt8tEOF6IV7GvUnD7CFVM40DMHkxn",
            to_absolute_path("../PACKETS/iglesias2.out"),
        )
        print("...Downlaod completed.......")
    else:
        pass

    Truee = np.genfromtxt(to_absolute_path("../PACKETS/iglesias2.out"), dtype="float")
    Truee = np.reshape(Truee, (-1,))

    Ne = 1
    ini = []
    inip = []
    inij = []
    injj = 500 * np.ones((1, 4))
    for i in range(Ne):
        at1 = rescale_linear(Truee, aay, bby)
        at2 = rescale_linear(Truee, mpor, hpor)
        ini.append(at1.reshape(-1, 1))
        inip.append(at2.reshape(-1, 1))
        inij.append(injj)
    ini = np.hstack(ini)
    inip = np.hstack(inip)
    kka = np.vstack(inij)

    start_time_plots1 = time.time()
    X_data2, data_use2 = inference_single(
        ini,
        inip,
        nx,
        ny,
        nz,
        max_t,
        DX,
        DY,
        DZ,
        BO,
        BW,
        CFL,
        timmee,
        MAXZ,
        factorr,
        steppi,
        LIR,
        UIR,
        LUB,
        HUB,
        aay,
        bby,
        mpor,
        hpor,
        dt,
        IWSw,
        PB,
        PATM,
        CFO,
        method,
        SWI,
        SWR,
        UW,
        UO,
        typee,
        step2,
        pini_alt,
        input_channel,
        kka,
    )
    elapsed_time_secs = time.time() - start_time_plots1
    msg = "FVM Reservoir simulation  took: %s secs (Wall clock time)" % timedelta(
        seconds=round(elapsed_time_secs)
    )
    print(msg)
    print("")
    print("")
    print("Finished FVM simulation")

    ini_ensemble1 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble2 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble3 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble4 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble5 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble6 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble7 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)

    ini_ensemble8 = np.zeros((Ne, steppi, nx, ny))  # Pressure
    ini_ensemble9 = np.zeros((Ne, steppi, nx, ny))  # Water saturation

    for kk in range(X_data2.shape[0]):
        perm = X_data2[kk, 0, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble1[kk, :, :, :] = permin

        perm = X_data2[kk, 1, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble2[kk, :, :, :] = permin

        perm = X_data2[kk, 2, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble3[kk, :, :, :] = permin

        perm = X_data2[kk, 3, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble4[kk, :, :, :] = permin

        perm = X_data2[kk, 4, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble5[kk, :, :, :] = permin

        perm = X_data2[kk, 5, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble6[kk, :, :, :] = permin

        perm = X_data2[kk, 6, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble7[kk, :, :, :] = permin

        perm = data_use2[kk, :steppi, :, :]
        ini_ensemble8[kk, :, :, :] = np.clip(perm, 1 / pini_alt, 1.0)

        perm = data_use2[kk, steppi:, :, :]
        ini_ensemble9[kk, :, :, :] = perm

    sio.savemat(
        to_absolute_path("../PACKETS/single.mat"),
        {
            "perm": ini_ensemble1,
            "Q": ini_ensemble2,
            "Qw": ini_ensemble3,
            "Phi": ini_ensemble4,
            "Time": ini_ensemble5,
            "Pini": ini_ensemble6,
            "Swini": ini_ensemble7,
            "pressure": ini_ensemble8,
            "water_sat": ini_ensemble9,
        },
    )

    preprocess_FNO_mat(to_absolute_path("../PACKETS/single.mat"))

    # load training/ test data
    input_keys = [
        Key("perm", scale=(3.46327e-01, 3.53179e-01)),
        Key("Q", scale=(1.94683e-03, 3.70558e-02)),
        Key("Qw", scale=(2.03866e-03, 3.70199e-02)),
        Key("Phi", scale=(1.73163e-01, 1.76590e-01)),
        Key("Time", scale=(1.66667e-02, 7.45058e-09)),
        Key("Pini", scale=(1.00000e00, 0.00000e00)),
        Key("Swini", scale=(2.00000e-01, 4.91738e-07)),
    ]
    output_keys_pressure = [Key("pressure", scale=(2.87008e-01, 1.85386e-01))]

    output_keys_saturation = [Key("water_sat", scale=(3.12903e-01, 1.79786e-01))]

    input_keys1 = [
        Key("perm"),
        Key("Q"),
        Key("Qw"),
        Key("Phi"),
        Key("Time"),
        Key("Pini"),
        Key("Swini"),
    ]
    output_keys_pressure1 = [Key("pressure")]

    output_keys_saturation1 = [Key("water_sat")]

    invar_train, outvar_train_pressure, outvar_train_saturation = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulations.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_saturation],
        n_examples=cfg.custom.ntrain,
    )
    invar_test, outvar_test_pressure, outvar_test_saturation = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulations.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_saturation],
        n_examples=cfg.custom.ntest,
    )

    invar_train1, outvar_train1_pressure, outvar_train1_saturation = load_FNO_dataset2(
        to_absolute_path("../PACKETS/single.hdf5"),
        [k.name for k in input_keys1],
        [k.name for k in output_keys_pressure1],
        [k.name for k in output_keys_saturation1],
        n_examples=1,
    )

    # add additional constraining values for darcy variable
    outvar_train_pressure["pressured"] = np.zeros_like(
        outvar_train_pressure["pressure"]
    )
    outvar_train_saturation["saturationd"] = np.zeros_like(
        outvar_train_saturation["water_sat"]
    )

    outvar_train1_pressure["pressured"] = np.zeros_like(
        outvar_train1_pressure["pressure"]
    )
    outvar_train1_saturation["saturationd"] = np.zeros_like(
        outvar_train1_saturation["water_sat"]
    )

    train_dataset_pressure = DictGridDataset(invar_train, outvar_train_pressure)
    train_dataset_saturation = DictGridDataset(invar_train, outvar_train_saturation)

    train_dataset1_pressure = DictGridDataset(invar_train1, outvar_train1_pressure)
    train_dataset1_saturation = DictGridDataset(invar_train1, outvar_train1_saturation)

    # [init-node]

    # Define AFNO model for forward model (pressure)
    afno_pressure = AFNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        [Key("pressure", size=steppi)],
        (nx, ny),
        patch_size=3,
    )

    # Define AFNO model for forward model (saturation)
    afno_saturation = AFNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        [Key("water_sat", size=steppi)],
        (nx, ny),
        patch_size=3,
    )

    if approach == 3:
        derivatives = [
            Key("pressure", derivatives=[Key("x")]),
            Key("pressure", derivatives=[Key("y")]),
            Key("pressure", derivatives=[Key("x"), Key("x")]),
            Key("pressure", derivatives=[Key("y"), Key("y")]),
        ]

        afno_pressure.add_pino_gradients(
            derivatives=derivatives,
            domain_length=[nx, ny],
        )

    inputs = [
        "perm",
        "Q",
        "Qw",
        "Phi",
        "Time",
        "Pini",
        "Swini",
        "pressure",
        "water_sat",
    ]

    if approach == 3:
        inputs += [
            "pressure__x",
            "pressure__y",
        ]

    darcyy = Node(
        inputs=inputs,
        outputs=[
            "pressured",
            "saturationd",
        ],
        evaluate=Black_oil(
            UIR,
            pini_alt,
            LUB,
            HUB,
            aay,
            bby,
            SWI,
            SWR,
            UW,
            BW,
            UO,
            BO,
            MAXZ,
            nx,
            ny,
            approach,
        ),
        name="Darcy node",
    )
    nodes = (
        [darcyy]
        + [afno_pressure.make_node("afnop_forward_model_pressure", jit=cfg.jit)]
        + [afno_saturation.make_node("afnop_forward_model_saturation", jit=cfg.jit)]
    )

    # [constraint]
    # make domain
    domain = Domain()

    # add constraints to domain
    supervised_pressure = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset_pressure,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised_pressure, "supervised_pressure")

    supervised_saturation = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset_saturation,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised_saturation, "supervised_saturation")

    supervised_pressure1 = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset1_pressure,
        batch_size=1,
    )
    domain.add_constraint(supervised_pressure1, "supervised_pressure1")

    supervised_saturation1 = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset1_saturation,
        batch_size=1,
    )
    domain.add_constraint(supervised_saturation1, "supervised_saturation1")

    # [constraint]
    # add validator

    test_pressure = GridValidator(
        nodes,
        dataset=train_dataset1_pressure,
        batch_size=1,
        plotter=CustomValidatorPlotterP(
            timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt
        ),
        requires_grad=False,
    )
    domain.add_validator(test_pressure, "test_pressure")

    test_saturation = GridValidator(
        nodes,
        dataset=train_dataset1_saturation,
        batch_size=1,
        plotter=CustomValidatorPlotterS(
            timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt
        ),
        requires_grad=False,
    )
    domain.add_validator(test_saturation, "test_saturation")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
