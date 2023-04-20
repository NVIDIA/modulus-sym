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
from sympy import Symbol, sin

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain.constraint import (
    PointwiseConstraint,
)
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node
from wave_equation import WaveEquation1D


@modulus.sym.main(config_path="conf", config_name="config_inverse")
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    we = WaveEquation1D(c="c")
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    invert_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("c")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        we.make_nodes(detach_names=["u__x", "u__x__x", "u__t__t"])
        + [wave_net.make_node(name="wave_network")]
        + [invert_net.make_node(name="invert_network")]
    )

    # prepare input data
    L = float(np.pi)
    deltaT = 0.01
    deltaX = 0.01
    x = np.arange(0, L, deltaX)
    t = np.arange(0, 2 * L, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    u = np.sin(X) * (np.cos(T) + np.sin(T))
    invar_numpy = {"x": X, "t": T}
    outvar_numpy = {"u": u}
    outvar_numpy["wave_equation"] = np.zeros_like(outvar_numpy["u"])

    # add constraints to solver

    # make domain
    domain = Domain()

    # data and pde loss
    data = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=invar_numpy,
        outvar=outvar_numpy,
        batch_size=cfg.batch_size.data,
    )
    domain.add_constraint(data, "interior_data")

    # add monitors
    monitor = PointwiseMonitor(
        invar_numpy,
        output_names=["c"],
        metrics={"mean_c": lambda var: torch.mean(var["c"])},
        nodes=nodes,
    )
    domain.add_monitor(monitor)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
