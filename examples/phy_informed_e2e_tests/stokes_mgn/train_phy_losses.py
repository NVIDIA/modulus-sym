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

import torch
import json
from sympy import Symbol, Function, Number
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import time, os
import wandb

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

try:
    import apex
except:
    pass

from modulus.models.meshgraphnet import MeshGraphNet
from modulus.datapipes.gnn.stokes_dataset import StokesDataset
from modulus.distributed.manager import DistributedManager

from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from utils import relative_lp_error

from modulus.sym.eq.phy_informer import PhysicsInformer
from modulus.sym.eq.pde import PDE
from modulus.sym.node import Node
from modulus.sym.key import Key


class Stokes(PDE):
    """Incompressible Stokes Flow"""

    def __init__(self, nu, dim=3):
        # set params
        self.dim = dim

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 2:
            input_variables.pop("z")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # set equations
        self.equations = {}
        self.equations["continuity"] = u.diff(x) + v.diff(y) + w.diff(z)
        self.equations["momentum_x"] = +p.diff(x) - nu * (
            u.diff(x).diff(x) + u.diff(y).diff(y) + u.diff(z).diff(z)
        )
        self.equations["momentum_y"] = +p.diff(y) - nu * (
            v.diff(x).diff(x) + v.diff(y).diff(y) + v.diff(z).diff(z)
        )
        self.equations["momentum_z"] = +p.diff(z) - nu * (
            w.diff(x).diff(x) + w.diff(y).diff(y) + w.diff(z).diff(z)
        )

        if self.dim == 2:
            self.equations.pop("momentum_z")


class MGNTrainer:
    def __init__(self, cfg: DictConfig, dist, rank_zero_logger):
        self.dist = dist
        self.rank_zero_logger = rank_zero_logger
        self.amp = cfg.amp

        # instantiate dataset
        self.dataset = StokesDataset(
            name="stokes_train",
            data_dir=to_absolute_path(cfg.data_dir),
            split="train",
            num_samples=cfg.num_training_samples,
        )

        # instantiate validation dataset
        self.validation_dataset = StokesDataset(
            name="stokes_validation",
            data_dir=to_absolute_path(cfg.data_dir),
            split="validation",
            num_samples=cfg.num_validation_samples,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )

        # instantiate validation dataloader
        self.validation_dataloader = GraphDataLoader(
            self.validation_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
        )

        # read the mu and std generated
        with open(to_absolute_path("./outputs/node_stats.json"), "r") as f:
            json_data = json.load(f)

        u_mean = json_data["u_mean"]
        v_mean = json_data["v_mean"]
        p_mean = json_data["p_mean"]
        u_std = json_data["u_std"]
        v_std = json_data["v_std"]
        p_std = json_data["p_std"]
        self.mu = torch.tensor([u_mean, v_mean, p_mean]).reshape(1, -1).to(dist.device)
        self.std = torch.tensor([u_std, v_std, p_std]).reshape(1, -1).to(dist.device)

        # Setup Phy informer
        stokes_pde = Stokes(nu=0.01, dim=2)
        self.phy_informer = PhysicsInformer(
            required_outputs=["continuity", "momentum_x", "momentum_y"],
            equations=stokes_pde,
            grad_method="least_squares",
            device=dist.device,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            cfg.input_dim_nodes,
            cfg.input_dim_edges,
            cfg.output_dim,
            aggregation=cfg.aggregation,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
        )
        if cfg.jit:
            self.model = torch.jit.script(self.model).to(dist.device)
        else:
            self.model = self.model.to(dist.device)

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()
        try:
            self.optimizer = apex.optimizers.FusedAdam(
                self.model.parameters(), lr=cfg.lr
            )
            rank_zero_logger.info("Using FusedAdam optimizer")
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def train(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
        nodes = graph.nodes().reshape(-1, 1)
        edges = torch.stack([graph.edges()[0], graph.edges()[1]], dim=1)
        coords = graph.ndata["pos"]
        # coords = torch.cat([coords, torch.zeros((coords.shape[0], 1)).to(coords.device)], dim=1)

        loss_data = self.criterion(pred, graph.ndata["y"])
        pred = self.dataset.denormalize(pred, self.mu, self.std)

        mask_interior = graph.ndata["marker"][:, 0] == 1
        mask_inflow = graph.ndata["marker"][:, 1] == 1
        mask_outflow = graph.ndata["marker"][:, 2] == 1
        mask_wall = graph.ndata["marker"][:, 3] == 1
        mask_poly = graph.ndata["marker"][:, 4] == 1

        loss_dict = self.phy_informer.forward(
            {
                "coordinates": coords,
                "nodes": nodes,
                "edges": edges,
                "u": pred[:, 0:1],
                "v": pred[:, 1:2],
                "p": pred[:, 2:3],
            }
        )

        continuity = loss_dict["continuity"][mask_interior]
        momentum_x = loss_dict["momentum_x"][mask_interior]
        momentum_y = loss_dict["momentum_y"][mask_interior]

        inflow_u_true = 4 * 0.3 * coords[:, 1:2] * (0.4 - coords[:, 1:2]) / 0.4 / 0.4
        inflow_u_true = inflow_u_true[mask_inflow]
        integral_flow = 4 * 0.3 * 0.4 / 6

        outflow_p = pred[:, 2:3][mask_outflow]
        outflow_u = pred[:, 0:1][mask_outflow]
        inflow_u = pred[:, 0:1][mask_inflow]
        inflow_v = pred[:, 1:2][mask_inflow]
        noslip_u = pred[:, 0:1][mask_wall]
        noslip_v = pred[:, 1:2][mask_wall]
        poly_u = pred[:, 0:1][mask_poly]
        poly_v = pred[:, 1:2][mask_poly]

        loss_continuity = torch.mean(continuity**2)
        loss_momentum_x = torch.mean(momentum_x**2)
        loss_momentum_y = torch.mean(momentum_y**2)
        loss_outflow_p = torch.mean(outflow_p**2)
        loss_inflow_u = torch.mean((inflow_u - inflow_u_true) ** 2)
        loss_inflow_v = torch.mean(inflow_v**2)
        loss_noslip_u = torch.mean(noslip_u**2)
        loss_noslip_v = torch.mean(noslip_v**2)
        loss_poly_u = torch.mean(poly_u**2)
        loss_poly_v = torch.mean(poly_v**2)
        loss_ic = (torch.mean(outflow_u) - 0.2) ** 2

        phy_loss = (
            10 * loss_inflow_u
            + 10 * loss_inflow_v
            + 10 * (loss_noslip_u + 1 * loss_poly_u)
            + 10 * (loss_noslip_v + 1 * loss_poly_v)
            + 10 * loss_continuity
            + loss_momentum_x
            + loss_momentum_y
            + loss_outflow_p
        )
        loss = loss_data + 0.01 * phy_loss
        self.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=self.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            loss = self.criterion(pred, graph.ndata["y"])
            return loss

    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        lr = self.get_lr()
        wandb.log({"lr": lr})

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    @torch.no_grad()
    def validation(self):
        error_keys = ["u", "v", "p"]
        errors = {key: 0 for key in error_keys}

        for graph in self.validation_dataloader:
            graph = graph.to(self.dist.device)
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)

            for index, key in enumerate(error_keys):
                pred_val = pred[:, index : index + 1]
                target_val = graph.ndata["y"][:, index : index + 1]
                errors[key] += relative_lp_error(pred_val, target_val)

        for key in error_keys:
            errors[key] = errors[key] / len(self.validation_dataloader)
            self.rank_zero_logger.info(f"validation error_{key} (%): {errors[key]}")

        wandb.log(
            {
                "val_u_error (%)": errors["u"],
                "val_v_error (%)": errors["v"],
                "val_p_error (%)": errors["p"],
            }
        )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize loggers
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name="Stokes-Training",
        group="Stokes-DDP-Group",
        mode=cfg.wandb_mode,
    )

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        loss_agg = 0
        for graph in trainer.dataloader:
            loss = trainer.train(graph)
            loss_agg += loss.detach().cpu().numpy()
        loss_agg /= len(trainer.dataloader)
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {loss_agg:10.3e}, lr: {trainer.get_lr()}, time per epoch: {(time.time() - start):10.3e}"
        )
        wandb.log({"loss": loss_agg})

        # validation
        if dist.rank == 0:
            trainer.validation()

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Saved model on rank {dist.rank}")
            start = time.time()
    rank_zero_logger.info("Training completed!")


if __name__ == "__main__":
    main()
