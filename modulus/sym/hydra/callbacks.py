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

"""
Supported Modulus callback configs
"""

import torch
import logging

from dataclasses import dataclass
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from hydra.experimental.callback import Callback
from typing import Any
from omegaconf import DictConfig, OmegaConf

from modulus.sym.distributed import DistributedManager
from modulus.sym.manager import JitManager, JitArchMode, GraphManager

logger = logging.getLogger(__name__)


class ModulusCallback(Callback):
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        # Update dist manager singleton with config parameters
        manager = DistributedManager()
        manager.broadcast_buffers = config.broadcast_buffers
        manager.find_unused_parameters = config.find_unused_parameters
        manager.cuda_graphs = config.cuda_graphs

        # jit manager
        jit_manager = JitManager()
        jit_manager.init(
            config.jit,
            config.jit_arch_mode,
            config.jit_use_nvfuser,
            config.jit_autograd_nodes,
        )

        # graph manager
        graph_manager = GraphManager()
        graph_manager.init(
            config.graph.func_arch,
            config.graph.func_arch_allow_partial_hessian,
            config.debug,
        )
        # The FuncArch does not work with TorchScript at all, so we raise
        # a warning and disabled it.
        if config.graph.func_arch and jit_manager.enabled:
            jit_manager.enabled = False
            logger.warning("Disabling JIT because functorch does not work with it.")

        logger.info(jit_manager)
        logger.info(graph_manager)


DefaultCallbackConfigs = DictConfig(
    {
        "modulus_callback": OmegaConf.create(
            {
                "_target_": "modulus.sym.hydra.callbacks.ModulusCallback",
            }
        )
    }
)


def register_callbacks_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="hydra/callbacks",
        name="default_callback",
        node=DefaultCallbackConfigs,
    )
