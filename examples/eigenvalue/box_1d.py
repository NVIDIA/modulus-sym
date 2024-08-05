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
One-dimensional particle in a box.
This example demonstrates how to use Modulus to solve the Schrödinger equation
for a particle in a box. The particle is fully confined; zero potential inside the
box and infinite potential outside. The goal is to find the eigenvalues and eigenfunctions.

Implementation based on:
Physics-Informed Neural Networks for Quantum Eigenvalue Problems
Henry Jin, Marios Mattheakis, Pavlos Protopapas
https://arxiv.org/abs/2203.00451
"""

import json
import os
import traceback

import modulus
import numpy as np

import torch.nn

from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.hydra import ModulusConfig, instantiate_arch
from modulus.sym.domain import Domain
from modulus.sym import Key, Node
from modulus.sym.solver import Solver
from modulus.sym.loss.loss import PointwiseLossNorm, IntegralLossNorm

from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.inferencer import PointwiseInferencer


from box_ode import BoxPDE
from custom import (
    SingleValue,
    InferencerPlotterCustom,
    IntegralInteriorConstraint,
    ProductModule,
    build_orthogonal_function_nodes,
)


def set_constraints(cfg, all_nodes, geo, domain):
    batch_size = cfg.batch_size

    # Assume the wavefunction is zero outside the domain
    # So set it to be 0 at the boundary for continuity.
    # Need to weight this fairly heavily.
    boundary_continuity = PointwiseBoundaryConstraint(
        nodes=all_nodes,
        geometry=geo,
        outvar={"psi": 0},
        lambda_weighting={"psi": 1.0},
        loss=PointwiseLossNorm(ord=2),
        batch_size=batch_size,
    )
    domain.add_constraint(boundary_continuity, name="boundary_continuity")

    # Psi = 0 will generally satisfy the equation, need to guard against that
    norm_constraint = IntegralInteriorConstraint(
        nodes=all_nodes,
        geometry=geo,
        outvar={"psi_norm": 1.0},
        lambda_weighting={"psi_norm": 1.0 / cfg.custom.integral_batch_size},
        loss=IntegralLossNorm(ord=2),
        batch_size=batch_size,
        integral_batch_size=cfg.custom.integral_batch_size,
    )
    domain.add_constraint(norm_constraint, name="norm_constraint")

    # Must satisfy the Schrödinger equation
    schrodinger_eqn_constraint = PointwiseInteriorConstraint(
        nodes=all_nodes,
        geometry=geo,
        outvar={"sch_equation": 0},
        lambda_weighting={"sch_equation": 1.0e-2},
        loss=PointwiseLossNorm(ord=2),
        batch_size=batch_size,
    )
    domain.add_constraint(schrodinger_eqn_constraint, name="schrodinger_eqn_constraint")

    # Avoid the trivial solution of E = 0.
    # This seems to happen even with nonzero-phi.
    nonzero_E = PointwiseBoundaryConstraint(
        nodes=all_nodes,
        geometry=geo,
        outvar={"E_inv": 0},
        lambda_weighting={"E_inv": 0.1e-2},
        loss=PointwiseLossNorm(ord=2),
        batch_size=batch_size,
    )
    domain.add_constraint(nonzero_E, name="nonzero_E")

    if cfg.custom.l_drive_weight > 0:
        # Tune e_const to set a minimum value for the eigenvalue,
        # so we can find multiple eigenvalues.
        drive_constraint = PointwiseInteriorConstraint(
            nodes=all_nodes,
            geometry=geo,
            outvar={"L_drive": 0},
            lambda_weighting={"L_drive": cfg.custom.l_drive_weight},
            loss=PointwiseLossNorm(ord=1),
            batch_size=batch_size,
        )
        domain.add_constraint(drive_constraint, name="drive_constraint")


def set_orthogonal_constraints(cfg, all_nodes, geo, domain, orth_func_nodes):
    """
    Orthogonality  constraint.

    For finding multiple eigenfunctions, we can impose an orthogonality constraint
    with respect to a known function. This ensures we don't arrive at a solution
    we've already found. `orth_func_nodes` are the nodes representing the known functions.
    """
    batch_size = cfg.batch_size
    num_nodes = len(orth_func_nodes)
    # Set this so the total orthogonality weighting loss is split evenly
    # among the functions.
    weighting_factor = 10.0 / (num_nodes * cfg.custom.integral_batch_size)
    outvar = {node.name: 0.0 for node in orth_func_nodes}
    lambda_weighting = {node.name: weighting_factor for node in orth_func_nodes}
    orthogonal_function = IntegralInteriorConstraint(
        nodes=all_nodes,
        geometry=geo,
        outvar=outvar,
        lambda_weighting=lambda_weighting,
        loss=IntegralLossNorm(ord=2),
        batch_size=batch_size,
        integral_batch_size=cfg.custom.integral_batch_size,
    )
    domain.add_constraint(orthogonal_function, name="orthogonal_function")


def set_monitors(cfg, all_nodes, geo: Line1D, domain):
    n_points = 100

    bound_ranges = list(geo.bounds.bound_ranges.values())[0]
    box_width = bound_ranges[1] - bound_ranges[0]
    dx = box_width / n_points

    interior_points = geo.sample_interior(n_points)
    interior_points = {"x": np.sort(interior_points["x"], axis=0)}

    box_region_monitor = PointwiseMonitor(
        invar=interior_points,
        output_names=["E", "psi_norm"],
        metrics={
            "E": lambda var: var["E"].mean(),
            "psi_norm": lambda var: ((var["psi"].abs()) ** 2 * dx).sum(),
        },
        nodes=all_nodes,
    )
    domain.add_monitor(box_region_monitor, name="box_region_monitor")

    box_boundary_monitor = PointwiseMonitor(
        invar=geo.sample_boundary(n_points),
        output_names=["psi"],
        metrics={"psi_at_boundary": lambda var: var["psi"].mean()},
        nodes=all_nodes,
    )
    domain.add_monitor(box_boundary_monitor, name="box_boundary_monitor")

    psi_norm_inference = PointwiseInferencer(
        nodes=all_nodes,
        invar=interior_points,
        output_names=["psi"],
        plotter=InferencerPlotterCustom(),
    )
    domain.add_inferencer(psi_norm_inference, name="psi")

    psi_norm_inference = PointwiseInferencer(
        nodes=all_nodes,
        invar=interior_points,
        output_names=["psi_norm"],
        plotter=InferencerPlotterCustom(),
    )
    domain.add_inferencer(psi_norm_inference, name="psi_norm")


def plot_results(result_dict, output_dir):
    from matplotlib import pyplot as plt

    psi_squared = result_dict["psi_squared"]
    x = result_dict["x"]
    E = float(np.mean(result_dict["E"]))
    psi = result_dict["psi"]
    xmin, xmax = np.min(x), np.max(x)

    # Plot psi^2
    plt.figure()
    plt.plot(x, psi_squared, label="Psi^2")
    _ = plt.title("$|\psi|^2$, E={:0.4e}".format(E))
    plt.savefig(os.path.join(output_dir, "psi_squared.png"))

    # Plot psi
    plt.figure()
    plt.plot(x, psi, label="Psi")
    _ = plt.title("$\psi$, E={:0.4e}".format(E))
    plt.savefig(os.path.join(output_dir, "psi.png"))


def calculate_sample_outputs(cfg, sch_net=None, E_value=0.0, model_dir=None):
    import torch

    if sch_net is None:
        assert model_dir is not None, "Must provide model_dir if no sch_net is provided"
        model_path = os.path.join(model_dir, "sch_net.0.pth")
        sch_net, _, _ = build_model(cfg)
        obj = torch.load(model_path)
        _ = sch_net.load_state_dict(obj)

    _ = sch_net.eval()
    eval_device = "cpu"
    sch_net.to(eval_device)

    xmin = 0.0
    xmax = 1.0
    with torch.inference_mode():
        x_tensor = torch.linspace(xmin, xmax, 100).reshape(-1, 1)
        x = x_tensor.numpy().flatten()
        data_out = sch_net({"x": x_tensor.to(eval_device)})
        psi = data_out["psi"].numpy()
        E = E_value

    dx = x[1] - x[0]
    psi_squared = np.abs(psi) ** 2
    psi_norm = np.sum(psi_squared * dx)

    return {
        "x": x.tolist(),
        "psi": psi.tolist(),
        "E": E,
        "psi_squared": psi_squared.tolist(),
        "psi_norm": float(psi_norm),
    }


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(f"Current working directory: {os.getcwd()}")

    seed = getattr(cfg.custom, "seed", None)
    if seed is not None:
        print(f"Setting seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Include these so they can be varied / made parameters
    hbar = 1.0
    mass = 1.0
    box_width = 1.0
    domain_bounds = (0.0, box_width)

    sch_net, input_keys, output_keys = build_model(cfg)

    # Eigenvalue which we are searching for
    E = Key("E")
    E_net = SingleValue(cfg.custom.e_const, name="E")

    E_node = Node(
        inputs=input_keys, outputs=[E], evaluate=E_net, name="E", optimize=True
    )

    schrodinger_pde = BoxPDE(hbar=hbar, mass=mass, e_const=cfg.custom.e_const)
    # schrodinger_pde.pprint()

    geo = Line1D(*domain_bounds)
    box_region = Line1D(0, box_width)

    domain = Domain()

    # make list of nodes to unroll graph on
    pde_nodes = schrodinger_pde.make_nodes()
    sch_psi_node = sch_net.make_node(name="sch_net")

    # To find different eigenfunctions,
    # we impose an orthogonality constraint with respect to a known function.
    # I'm using different modes of the sine function, but
    # this could also be a previously trained network.
    mode_ns = getattr(cfg.custom, "orthogonal_modes", [])
    print(f"Orthogonal modes: {mode_ns}")
    mode_ns = mode_ns if mode_ns else []
    orth_func_nodes = build_orthogonal_function_nodes(
        input_keys, "psi", mode_ns, box_width
    )

    all_nodes = pde_nodes + [sch_psi_node, E_node] + orth_func_nodes

    set_constraints(cfg, all_nodes, geo, domain)
    set_monitors(cfg, all_nodes, geo, domain)

    if orth_func_nodes:
        set_orthogonal_constraints(cfg, all_nodes, geo, domain, orth_func_nodes)

    # Solve
    # Requires CUDA
    try:
        print(f"Beginning solver")
        slv = Solver(cfg, domain)
        slv.solve()
    except Exception as exc:
        print(f"Error occured during solving: {exc}\n.Traceback: \n")
        traceback.print_exc()

    output_dir = "local_box_1d"
    print(f"box_1d Run complete. Saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sch_net.save(output_dir)

    E_value = float(
        E_net({"x": torch.ones(1).to(E_net.get_device())})["E"]
        .detach()
        .cpu()
        .numpy()[0]
    )

    result_dict = calculate_sample_outputs(cfg, sch_net=sch_net, E_value=E_value)
    print(f"Psi norm: {result_dict['psi_norm']:0.4e}")
    print(f"Eigenvalue: {E_value:0.4e}")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result_dict, f)

    plot_results(result_dict, ".")


def build_model(cfg: ModulusConfig):
    x = Key("x")
    psi = Key("psi")

    input_keys = [x]
    output_keys = [psi]
    arch = cfg.arch
    arch_cfg = next(iter(arch.values()))
    sch_net = instantiate_arch(
        input_keys=input_keys, output_keys=output_keys, cfg=arch_cfg
    )

    return sch_net, input_keys, output_keys


if __name__ == "__main__":
    run()
