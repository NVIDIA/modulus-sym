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
from modulus.sym.models.fno import *
import shutil
import cupy as cp
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
    def __init__(
        self,
        timmee,
        max_t,
        MAXZ,
        pini_alt,
        nx,
        ny,
        nz,
        steppi,
        tc2,
        dt,
        injectors,
        producers,
        N_injw,
        N_pr,
    ):
        self.timmee = timmee
        self.max_t = max_t
        self.MAXZ = MAXZ
        self.pini_alt = pini_alt
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.steppi = steppi
        self.tc2 = tc2
        self.dt = dt
        self.injectors = injectors
        self.producers = producers
        self.N_injw = N_injw
        self.N_pr = N_pr

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables

        # get and interpolate output variable
        pressure_true, pressure_pred = true_outvar["pressure"], pred_outvar["pressure"]

        # make plot

        f_big = []
        Time_vector = np.zeros((self.steppi))

        Accuracy_presure = np.zeros((self.steppi, 2))
        for itt in range(self.steppi):
            Time_vector[itt] = int((itt + 1) * self.dt * self.MAXZ)
            look = (pressure_pred[0, itt, :, :]) * self.pini_alt
            lookf = (pressure_true[0, itt, :, :]) * self.pini_alt
            diff1 = abs(look - lookf)

            XX, YY = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
            f_2 = plt.figure(figsize=(10, 10), dpi=100)
            plt.subplot(1, 3, 1)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                look,
                self.N_injw,
                self.N_pr,
                "pressure Modulus",
                self.injectors,
                self.producers,
            )
            plt.subplot(1, 3, 2)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                lookf,
                self.N_injw,
                self.N_pr,
                "pressure Numerical",
                self.injectors,
                self.producers,
            )
            plt.subplot(1, 3, 3)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                diff1,
                self.N_injw,
                self.N_pr,
                "pressure diff",
                self.injectors,
                self.producers,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = "Timestep --" + str(int((itt + 1) * self.dt * self.MAXZ)) + " days"

            plt.suptitle(tita, fontsize=16)
            namez = "pressure_simulations" + str(int(itt))
            yes = (f_2, namez)
            f_big.append(yes)
            # plt.clf()
            plt.close()

            R2p, L2p = compute_metrics(look.ravel(), lookf.ravel())
            Accuracy_presure[itt, 0] = R2p
            Accuracy_presure[itt, 1] = L2p

            f_3 = plt.figure(figsize=(20, 20), dpi=200)
            ax1 = f_3.add_subplot(131, projection="3d")
            Plot_Modulus(
                ax1,
                self.nx,
                self.ny,
                self.nz,
                look,
                self.N_injw,
                self.N_pr,
                "pressure Modulus",
                self.injectors,
                self.producers,
            )
            ax2 = f_3.add_subplot(132, projection="3d")
            Plot_Modulus(
                ax2,
                self.nx,
                self.ny,
                self.nz,
                lookf,
                self.N_injw,
                self.N_pr,
                "pressure Numerical",
                self.injectors,
                self.producers,
            )
            ax3 = f_3.add_subplot(133, projection="3d")
            Plot_Modulus(
                ax3,
                self.nx,
                self.ny,
                self.nz,
                diff1,
                self.N_injw,
                self.N_pr,
                "pressure diff",
                self.injectors,
                self.producers,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = (
                "3D Map - Timestep --"
                + str(int((itt + 1) * self.dt * self.MAXZ))
                + " days"
            )

            plt.suptitle(tita, fontsize=20, weight="bold")

            namez = "Simulations3Dp" + str(int(itt))
            yes2 = (f_3, namez)
            f_big.append(yes2)
            # plt.clf()
            plt.close()

        fig4 = plt.figure(figsize=(10, 10), dpi=200)

        font = FontProperties()
        font.set_family("Helvetica")
        font.set_weight("bold")

        fig4.text(
            0.5,
            0.98,
            "R2(%) Accuracy - Modulus/Numerical(GPU)",
            ha="center",
            va="center",
            fontproperties=font,
            fontsize=16,
        )
        fig4.text(
            0.5,
            0.49,
            "L2(%) Accuracy - Modulus/Numerical(GPU)",
            ha="center",
            va="center",
            fontproperties=font,
            fontsize=16,
        )

        # Plot R2 accuracies
        plt.subplot(2, 1, 1)
        plt.plot(
            Time_vector,
            Accuracy_presure[:, 0],
            label="R2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("Pressure", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("R2(%)", fontproperties=font)

        # Plot L2 accuracies
        plt.subplot(2, 1, 2)
        plt.plot(
            Time_vector,
            Accuracy_presure[:, 1],
            label="L2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("Pressure", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("L2(%)", fontproperties=font)

        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        namez = "R2L2_pressure"
        yes21 = (fig4, namez)
        f_big.append(yes21)
        # plt.clf()
        plt.close()

        return f_big


class CustomValidatorPlotterS(ValidatorPlotter):
    def __init__(
        self,
        timmee,
        max_t,
        MAXZ,
        pini_alt,
        nx,
        ny,
        nz,
        steppi,
        tc2,
        dt,
        injectors,
        producers,
        N_injw,
        N_pr,
    ):
        self.timmee = timmee
        self.max_t = max_t
        self.MAXZ = MAXZ
        self.pini_alt = pini_alt
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.steppi = steppi
        self.tc2 = tc2
        self.dt = dt
        self.injectors = injectors
        self.producers = producers
        self.N_injw = N_injw
        self.N_pr = N_pr

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables

        water_true, water_pred = true_outvar["water_sat"], pred_outvar["water_sat"]

        # make plot

        f_big = []
        Accuracy_oil = np.zeros((self.steppi, 2))
        Accuracy_water = np.zeros((self.steppi, 2))
        Time_vector = np.zeros((self.steppi))
        for itt in range(self.steppi):
            Time_vector[itt] = int((itt + 1) * self.dt * self.MAXZ)

            XX, YY = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
            f_2 = plt.figure(figsize=(12, 12), dpi=100)

            look_sat = water_pred[0, itt, :, :]  # *1e-2
            look_oil = 1 - look_sat

            lookf_sat = water_true[0, itt, :, :]  # * 1e-2
            lookf_oil = 1 - lookf_sat

            diff1_wat = abs(look_sat - lookf_sat)
            diff1_oil = abs(look_oil - lookf_oil)

            plt.subplot(2, 3, 1)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                look_sat,
                self.N_injw,
                self.N_pr,
                "water Modulus",
                self.injectors,
                self.producers,
            )
            plt.subplot(2, 3, 2)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                lookf_sat,
                self.N_injw,
                self.N_pr,
                "water Numerical",
                self.injectors,
                self.producers,
            )
            plt.subplot(2, 3, 3)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                diff1_wat,
                self.N_injw,
                self.N_pr,
                "water diff",
                self.injectors,
                self.producers,
            )
            R2w, L2w = compute_metrics(look_sat.ravel(), lookf_sat.ravel())
            Accuracy_water[itt, 0] = R2w
            Accuracy_water[itt, 1] = L2w

            plt.subplot(2, 3, 4)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                look_oil,
                self.N_injw,
                self.N_pr,
                "oil Modulus",
                self.injectors,
                self.producers,
            )
            plt.subplot(2, 3, 5)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                lookf_oil,
                self.N_injw,
                self.N_pr,
                "oil Numerical",
                self.injectors,
                self.producers,
            )
            plt.subplot(2, 3, 6)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                diff1_oil,
                self.N_injw,
                self.N_pr,
                "oil diff",
                self.injectors,
                self.producers,
            )

            R2o, L2o = compute_metrics(look_oil.ravel(), lookf_oil.ravel())
            Accuracy_oil[itt, 0] = R2o
            Accuracy_oil[itt, 1] = L2o

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = "Timestep --" + str(int((itt + 1) * self.dt * self.MAXZ)) + " days"

            plt.suptitle(tita, fontsize=16)
            namez = "saturation_simulations" + str(int(itt))
            yes = (f_2, namez)
            f_big.append(yes)
            # plt.clf()
            plt.close()

            f_3 = plt.figure(figsize=(20, 20), dpi=200)
            ax1 = f_3.add_subplot(231, projection="3d")
            Plot_Modulus(
                ax1,
                self.nx,
                self.ny,
                self.nz,
                look_sat,
                self.N_injw,
                self.N_pr,
                "water Modulus",
                self.injectors,
                self.producers,
            )
            ax2 = f_3.add_subplot(232, projection="3d")
            Plot_Modulus(
                ax2,
                self.nx,
                self.ny,
                self.nz,
                lookf_sat,
                self.N_injw,
                self.N_pr,
                "water Numerical",
                self.injectors,
                self.producers,
            )
            ax3 = f_3.add_subplot(233, projection="3d")
            Plot_Modulus(
                ax3,
                self.nx,
                self.ny,
                self.nz,
                diff1_wat,
                self.N_injw,
                self.N_pr,
                "water diff",
                self.injectors,
                self.producers,
            )

            ax4 = f_3.add_subplot(234, projection="3d")
            Plot_Modulus(
                ax4,
                self.nx,
                self.ny,
                self.nz,
                look_oil,
                self.N_injw,
                self.N_pr,
                "oil Modulus",
                self.injectors,
                self.producers,
            )
            ax5 = f_3.add_subplot(235, projection="3d")
            Plot_Modulus(
                ax5,
                self.nx,
                self.ny,
                self.nz,
                lookf_oil,
                self.N_injw,
                self.N_pr,
                "oil Numerical",
                self.injectors,
                self.producers,
            )
            ax6 = f_3.add_subplot(236, projection="3d")
            Plot_Modulus(
                ax6,
                self.nx,
                self.ny,
                self.nz,
                diff1_oil,
                self.N_injw,
                self.N_pr,
                "oil diff",
                self.injectors,
                self.producers,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = (
                "3D Map - Timestep --"
                + str(int((itt + 1) * self.dt * self.MAXZ))
                + " days"
            )

            plt.suptitle(tita, fontsize=20, weight="bold")

            namez = "Simulations3Ds" + str(int(itt))
            yes2 = (f_3, namez)
            f_big.append(yes2)
            # plt.clf()
            plt.close()

        fig4 = plt.figure(figsize=(20, 20), dpi=200)

        font = FontProperties()
        font.set_family("Helvetica")
        font.set_weight("bold")

        fig4.text(
            0.5,
            0.98,
            "R2(%) Accuracy - Modulus/Numerical(GPU)",
            ha="center",
            va="center",
            fontproperties=font,
            fontsize=16,
        )
        fig4.text(
            0.5,
            0.49,
            "L2(%) Accuracy - Modulus/Numerical(GPU)",
            ha="center",
            va="center",
            fontproperties=font,
            fontsize=16,
        )

        plt.subplot(2, 2, 1)
        plt.plot(
            Time_vector,
            Accuracy_water[:, 0],
            label="R2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("water_saturation", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("R2(%)", fontproperties=font)

        plt.subplot(2, 2, 2)
        plt.plot(
            Time_vector,
            Accuracy_oil[:, 0],
            label="R2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("oil_saturation", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("R2(%)", fontproperties=font)

        plt.subplot(2, 2, 3)
        plt.plot(
            Time_vector,
            Accuracy_water[:, 1],
            label="L2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("water_saturation", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("L2(%)", fontproperties=font)

        plt.subplot(2, 2, 4)
        plt.plot(
            Time_vector,
            Accuracy_oil[:, 1],
            label="L2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("oil_saturation", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("L2(%)", fontproperties=font)

        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        namez = "R2L2_saturations"
        yes21 = (fig4, namez)
        f_big.append(yes21)
        # plt.clf()
        plt.close()

        return f_big


# [pde-loss]
# define custom class for black oil model
class Black_oil(torch.nn.Module):
    "Custom Black oil PDE definition for PINO"

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

            # krw = torch.square(S)
            # kroil = torch.square(torch.sub(torch.ones(S.shape,\
            #                         device = u.device),S))
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
    print("|                 TRAIN THE MODEL USING A 2D PINO APPROACH:        |")
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
    nx = cfg.custom.NVRS.nx
    ny = cfg.custom.NVRS.ny
    nz = cfg.custom.NVRS.nz
    BO = cfg.custom.NVRS.BO  # oil formation volume factor
    BW = cfg.custom.NVRS.BW  # Water formation volume factor
    UW = cfg.custom.NVRS.UW  # water viscosity in cP
    UO = cfg.custom.NVRS.UO  # oil viscosity in cP
    DX = cfg.custom.NVRS.DX  # size of pixel in x direction
    DY = cfg.custom.NVRS.DY  # sixze of pixel in y direction
    DZ = cfg.custom.NVRS.DZ  # sizze of pixel in z direction

    DX = cp.float32(DX)
    DY = cp.float32(DY)
    UW = cp.float32(UW)  # water viscosity in cP
    UO = cp.float32(UO)  # oil viscosity in cP
    SWI = cp.float32(cfg.custom.NVRS.SWI)
    SWR = cp.float32(cfg.custom.NVRS.SWR)
    pini_alt = cfg.custom.NVRS.pini_alt
    BW = cp.float32(BW)  # Water formation volume factor
    BO = cp.float32(BO)  # Oil formation volume factor

    # training
    LUB = cfg.custom.NVRS.LUB
    HUB = cfg.custom.NVRS.HUB  # Permeability rescale
    aay, bby = cfg.custom.NVRS.aay, cfg.custom.NVRS.bby  # Permeability range mD
    # Low_K, High_K = aay,bby

    # batch_size = cfg.custom.NVRS.batch_size #'size of simulated labelled data to run'
    timmee = (
        cfg.custom.NVRS.timmee
    )  # float(input ('Enter the time step interval duration for simulation (days): '))
    max_t = (
        cfg.custom.NVRS.max_t
    )  # float(input ('Enter the maximum time in days for simulation(days): '))
    MAXZ = cfg.custom.NVRS.MAXZ  # reference maximum time in days of simulation
    steppi = int(max_t / timmee)
    factorr = cfg.custom.NVRS.factorr  # from [0 1] excluding the limits for PermZ
    LIR = cfg.custom.NVRS.LIR  # lower injection rate
    UIR = cfg.custom.NVRS.UIR  # uppwer injection rate
    input_channel = (
        cfg.custom.NVRS.input_channel
    )  # [Perm, Q,QW,Phi,dt, initial_pressure, initial_water_sat]

    injectors = cfg.custom.WELLSPECS.water_injector_wells
    producers = cfg.custom.WELLSPECS.producer_wells
    N_injw = len(cfg.custom.WELLSPECS.water_injector_wells)  # Number of water injectors
    N_pr = len(cfg.custom.WELLSPECS.producer_wells)  # Number of producers

    # tc2 = Equivalent_time(timmee,2100,timmee,max_t)
    tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
    dt = np.diff(tc2)[0]  # Time-step

    bb = os.path.isfile(to_absolute_path("../PACKETS/Training4.mat"))
    if bb == False:
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
        print("Load simulated labelled training data from MAT file")
        matt = sio.loadmat(to_absolute_path("../PACKETS/Training4.mat"))

        X_data1 = matt["INPUT"]
        data_use1 = matt["OUTPUT"]

    bb = os.path.isfile(to_absolute_path("../PACKETS/Test4.mat"))
    if bb == False:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1G4Cvg8eIObyBK0eoo7iX-0hhMTnpJktj",
            to_absolute_path("../PACKETS/Test4.mat"),
        )
        print("...Downlaod completed.......")
        print("Load simulated labelled test data from MAT file")
        matt = sio.loadmat(to_absolute_path("../PACKETS/Test4.mat"))

        X_data2 = matt["INPUT"]
        data_use2 = matt["OUTPUT"]

    else:
        print("Load simulated labelled test data from MAT file")
        matt = sio.loadmat(to_absolute_path("../PACKETS/Test4.mat"))

        X_data2 = matt["INPUT"]
        data_use2 = matt["OUTPUT"]

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
        cPress[kk, :, :, :] = perm  # np.clip(perm ,1/pini_alt,1.)

        perm = data_use1[kk, steppi:, :, :]
        cSat[kk, :, :, :] = perm

    sio.savemat(
        to_absolute_path("../PACKETS/simulationstrain.mat"),
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

    preprocess_FNO_mat(to_absolute_path("../PACKETS/simulationstrain.mat"))

    cPerm = np.zeros((X_data2.shape[0], 1, nx, ny))  # Permeability
    cQ = np.zeros((X_data2.shape[0], 1, nx, ny))  # Overall source/sink term
    cQw = np.zeros((X_data2.shape[0], 1, nx, ny))  # Sink term
    cPhi = np.zeros((X_data2.shape[0], 1, nx, ny))  # Porosity
    cTime = np.zeros((X_data2.shape[0], 1, nx, ny))  # Time index
    cPini = np.zeros((X_data2.shape[0], 1, nx, ny))  # Initial pressure
    cSini = np.zeros((X_data2.shape[0], 1, nx, ny))  # Initial water saturation

    cPress = np.zeros((X_data2.shape[0], steppi, nx, ny))  # Pressure
    cSat = np.zeros((X_data2.shape[0], steppi, nx, ny))  # Water saturation

    for kk in range(X_data2.shape[0]):
        perm = X_data2[kk, 0, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPerm[kk, :, :, :] = permin

        perm = X_data2[kk, 1, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQ[kk, :, :, :] = permin

        perm = X_data2[kk, 2, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQw[kk, :, :, :] = permin

        perm = X_data2[kk, 3, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPhi[kk, :, :, :] = permin

        perm = X_data2[kk, 4, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cTime[kk, :, :, :] = permin

        perm = X_data2[kk, 5, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPini[kk, :, :, :] = permin

        perm = X_data2[kk, 6, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cSini[kk, :, :, :] = permin

        perm = data_use2[kk, :steppi, :, :]
        cPress[kk, :, :, :] = perm  # np.clip(perm ,1/pini_alt,1.)

        perm = data_use2[kk, steppi:, :, :]
        cSat[kk, :, :, :] = perm

    sio.savemat(
        to_absolute_path("../PACKETS/simulationstest.mat"),
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

    preprocess_FNO_mat(to_absolute_path("../PACKETS/simulationstest.mat"))

    # load training/ test data
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

    invar_train, outvar_train_pressure, outvar_train_saturation = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulationstrain.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_saturation],
        n_examples=cfg.custom.ntrain,
    )
    invar_test, outvar_test_pressure, outvar_test_saturation = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulationstest.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_saturation],
        n_examples=cfg.custom.ntest,
    )

    # add additional constraining values for darcy variable
    outvar_train_pressure["pressured"] = np.zeros_like(
        outvar_train_pressure["pressure"]
    )
    outvar_train_saturation["saturationd"] = np.zeros_like(
        outvar_train_saturation["water_sat"]
    )

    train_dataset_pressure = DictGridDataset(invar_train, outvar_train_pressure)
    train_dataset_saturation = DictGridDataset(invar_train, outvar_train_saturation)

    test_dataset_pressure = DictGridDataset(invar_test, outvar_test_pressure)
    test_dataset_saturation = DictGridDataset(invar_test, outvar_test_saturation)

    # [init-node]

    # Define FNO model for forward model (pressure)
    decoder1 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("pressure", size=steppi)]
    )
    fno_pressure = FNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        dimension=2,
        decoder_net=decoder1,
    )

    # Define FNO model for forward model (saturation)
    decoder2 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("water_sat", size=steppi)]
    )
    fno_saturation = FNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        dimension=2,
        decoder_net=decoder2,
    )

    # if approach ==3:
    #     derivatives = [
    #         Key("pressure", derivatives=[Key("x")]),
    #         Key("pressure", derivatives=[Key("y")]),
    #         Key("pressure", derivatives=[Key("x"), Key("x")]),
    #         Key("pressure", derivatives=[Key("y"), Key("y")]),
    #     ]

    #     fno_pressure.add_pino_gradients(
    #         derivatives=derivatives,
    #         domain_length=[nx, ny],
    #     )

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

    # if approach ==3:
    #     inputs += [
    #         "pressure__x",
    #         "pressure__y",
    #     ]

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
        + [fno_pressure.make_node("pino_forward_model_pressure", jit=cfg.jit)]
        + [fno_saturation.make_node("pino_forward_model_saturation", jit=cfg.jit)]
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

    # [constraint]
    # add validator

    # [constraint]
    # add validator
    # test_pressure = GridValidator(
    #     nodes,
    #     dataset=test_dataset_pressure,
    #     batch_size=1,
    #     plotter=CustomValidatorPlotterP(timmee,max_t,MAXZ,pini_alt,nx,ny,nz,\
    #                                 steppi,tc2,dt,injectors,producers,N_injw,N_pr),
    #     requires_grad=False,
    # )

    test_pressure = GridValidator(
        nodes,
        dataset=test_dataset_pressure,
        batch_size=1,
        requires_grad=False,
    )

    domain.add_validator(test_pressure, "test_pressure")

    # test_saturation = GridValidator(
    #     nodes,
    #     dataset=test_dataset_saturation,
    #     batch_size=1,
    #     plotter=CustomValidatorPlotterS(timmee,max_t,MAXZ,pini_alt,nx,ny,nz,\
    #                                 steppi,tc2,dt,injectors,producers,N_injw,N_pr),

    #     requires_grad=False,
    # )

    test_saturation = GridValidator(
        nodes,
        dataset=test_dataset_saturation,
        batch_size=1,
        requires_grad=False,
    )
    domain.add_validator(test_saturation, "test_saturation")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
