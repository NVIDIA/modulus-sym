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
from sympy import Symbol, Eq, sin, cos, Min, Max, Abs, log, exp, tanh

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from modulus.sym.utils.sympy.functions import parabola
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.node import Node

from custom_k_om import kOmegaInit, kOmega, kOmegaStdWF


@modulus.sym.main(config_path="conf_re590_k_om", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # simulation parameters
    Re = 590
    nu = 1 / Re
    y_plus = 30
    karman_constant = 0.4187
    resolved_y_start = y_plus * nu
    channel_width = (-1, 1)
    channel_length = (-np.pi / 2, np.pi / 2)

    inlet = Line(
        (channel_length[0], channel_width[0]),
        (channel_length[0], channel_width[1]),
        normal=1,
    )
    outlet = Line(
        (channel_length[1], channel_width[0]),
        (channel_length[1], channel_width[1]),
        normal=1,
    )

    geo_sdf = Channel2D(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )

    # geometry where the equations are solved
    geo_resolved = Channel2D(
        (channel_length[0], channel_width[0] + resolved_y_start),
        (channel_length[1], channel_width[1] - resolved_y_start),
    )

    # make list of nodes to unroll graph on
    init = kOmegaInit(nu=nu, rho=1.0)
    eq = kOmega(nu=nu, rho=1.0)
    wf = kOmegaStdWF(nu=nu, rho=1.0)

    u_tau_net = instantiate_arch(
        input_keys=[Key("u_in"), Key("y_in")],
        output_keys=[Key("u_tau_out")],
        cfg=cfg.arch.fully_connected,
    )
    flow_net = instantiate_arch(
        input_keys=[Key("x_sin"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    p_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("p")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    k_net = instantiate_arch(
        input_keys=[Key("x_sin"), Key("y")],
        output_keys=[Key("k_star")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    om_net = instantiate_arch(
        input_keys=[Key("x_sin"), Key("y")],
        output_keys=[Key("om_star")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )

    nodes = (
        init.make_nodes()
        + eq.make_nodes()
        + wf.make_nodes()
        + [
            Node.from_sympy(
                sin(2 * np.pi * Symbol("x") / (channel_length[1] - channel_length[0])),
                "x_sin",
            )
        ]
        + [Node.from_sympy(Min(log(1 + exp(Symbol("k_star"))) + 1e-4, 20), "k")]
        + [Node.from_sympy(Min(log(1 + exp(Symbol("om_star"))) + 1e-4, 20), "om_plus")]
        + [flow_net.make_node(name="flow_network")]
        + [p_net.make_node(name="p_network")]
        + [k_net.make_node(name="k_network")]
        + [om_net.make_node(name="om_network")]
    )

    nodes_u_tau = (
        [Node.from_sympy(Symbol("normal_distance"), "y_in")]
        + [
            Node.from_sympy(
                (
                    (
                        Symbol("u")
                        - (
                            Symbol("u") * (-Symbol("normal_x"))
                            + Symbol("v") * (-Symbol("normal_y"))
                        )
                        * (-Symbol("normal_x"))
                    )
                    ** 2
                    + (
                        Symbol("v")
                        - (
                            Symbol("u") * (-Symbol("normal_x"))
                            + Symbol("v") * (-Symbol("normal_y"))
                        )
                        * (-Symbol("normal_y"))
                    )
                    ** 2
                )
                ** 0.5,
                "u_parallel_to_wall",
            )
        ]
        + [Node.from_sympy(Symbol("u_parallel_to_wall"), "u_in")]
        + [Node.from_sympy(Symbol("u_tau_out"), "u_tau")]
        + [u_tau_net.make_node(name="u_tau_network", optimize=False)]
    )

    # add constraints to solver
    p_grad = 1.0

    x, y = Symbol("x"), Symbol("y")

    # make domain
    domain = Domain()

    # Point where wall funciton is applied
    wf_pt = PointwiseBoundaryConstraint(
        nodes=nodes + nodes_u_tau,
        geometry=geo_resolved,
        outvar={
            "velocity_wall_normal_wf": 0,
            "velocity_wall_parallel_wf": 0,
            "om_plus_wf": 0,
            "k_wf": 0,
            "wall_shear_stress_x_wf": 0,
            "wall_shear_stress_y_wf": 0,
        },
        lambda_weighting={
            "velocity_wall_normal_wf": 100,
            "velocity_wall_parallel_wf": 100,
            "om_plus_wf": 10,
            "k_wf": 1,
            "wall_shear_stress_x_wf": 100,
            "wall_shear_stress_y_wf": 100,
        },
        batch_size=cfg.batch_size.wf_pt,
        parameterization={"normal_distance": resolved_y_start},
    )
    domain.add_constraint(wf_pt, "WF")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_resolved,
        outvar={
            "continuity": 0,
            "momentum_x": 0,
            "momentum_y": 0,
            "k_equation": 0,
            "om_plus_equation": 0,
        },
        lambda_weighting={
            "continuity": 100,
            "momentum_x": 1000,
            "momentum_y": 1000,
            "k_equation": 10,
            "om_plus_equation": 0.1,
        },
        batch_size=cfg.batch_size.interior,
        bounds={x: channel_length, y: channel_width},
    )
    domain.add_constraint(interior, "Interior")

    # pressure pc
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"p": p_grad * (channel_length[1] - channel_length[0])},
        lambda_weighting={"p": 10},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "Inlet")

    # pressure pc
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        lambda_weighting={"p": 10},
        batch_size=cfg.batch_size.outlet,
    )
    domain.add_constraint(outlet, "Outlet")

    # flow initialization
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_resolved,
        outvar={"u_init": 0, "v_init": 0, "k_init": 0, "p_init": 0, "om_plus_init": 0},
        batch_size=cfg.batch_size.interior_init,
        bounds={x: channel_length, y: channel_width},
    )
    domain.add_constraint(interior, "InteriorInit")

    # add inferencing and monitor
    invar_wf_pt = geo_resolved.sample_boundary(
        1024, parameterization={"normal_distance": resolved_y_start}
    )
    u_tau_monitor = PointwiseMonitor(
        invar_wf_pt,
        output_names=["u_tau"],
        metrics={"mean_u_tau": lambda var: torch.mean(var["u_tau"])},
        nodes=nodes + nodes_u_tau,
    )
    domain.add_monitor(u_tau_monitor)

    # add inferencer data
    inference = PointwiseInferencer(
        nodes=nodes,
        invar=geo_resolved.sample_interior(
            5000, bounds={x: channel_length, y: channel_width}
        ),
        output_names=["u", "v", "p", "k", "om_plus"],
    )
    domain.add_inferencer(inference, "inf_interior")

    inference = PointwiseInferencer(
        nodes=nodes + nodes_u_tau,
        invar=geo_resolved.sample_boundary(
            10, parameterization={"normal_distance": resolved_y_start}
        ),
        output_names=["u", "v", "p", "k", "om_plus", "normal_distance", "u_tau"],
    )
    domain.add_inferencer(inference, "inf_wf")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
