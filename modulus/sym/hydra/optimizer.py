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
Supported optimizer configs
"""

import paddle

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Any
from omegaconf import MISSING


@dataclass
class OptimizerConf:
    _target_ = MISSING
    _params_: Any = field(
        default_factory=lambda: {
            "compute_gradients": "adam_compute_gradients",
            "apply_gradients": "adam_apply_gradients",
        }
    )


@dataclass
class AdamConf(OptimizerConf):
    _target_: str = "paddle.optimizer.Adam"
    learning_rate: Any = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-08
    weight_decay: float = 0


@dataclass
class SGDConf(OptimizerConf):
    _target_: str = "paddle.optimizer.SGD"
    learning_rate: float = 1.0e-3
    momentum: float = 1.0e-2
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


@dataclass
class BFGSConf(OptimizerConf):
    _target_: str = "paddle.optimizer.LBFGS"
    learning_rate: float = 1.0
    max_iter: int = 1000
    max_eval: Any = None
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: Any = None  # Union[None, str]
    _params_: Any = field(
        default_factory=lambda: {
            "compute_gradients": "bfgs_compute_gradients",
            "apply_gradients": "bfgs_apply_gradients",
        }
    )


@dataclass
class AdadeltaConf(OptimizerConf):
    _target_: str = "paddle.optimizer.Adadelta"
    learning_rate: float = 1.0
    rho: float = 0.9
    epsilon: float = 1e-6
    weight_decay: float = 0


@dataclass
class AdagradConf(OptimizerConf):
    _target_: str = "paddle.optimizer.Adagrad"
    learning_rate: float = 1.0e-2
    weight_decay: float = 0
    initial_accumulator_value: float = 0
    epsilon: float = 1e-10


@dataclass
class AdamWConf(OptimizerConf):
    _target_: str = "paddle.optimizer.AdamW"
    learning_rate: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False


@dataclass
class AdamaxConf(OptimizerConf):
    _target_: str = "paddle.optimizer.Adamax"
    learning_rate: float = 2.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0


@dataclass
class RMSpropConf(OptimizerConf):
    _target_: str = "paddle.optimizer.RMSprop"
    learning_rate: float = 1.0e-2
    alpha: float = 0.99
    epsilon: float = 1e-8
    weight_decay: float = 0
    momentum: float = 0
    centered: bool = False


@dataclass
class LambConf(OptimizerConf):
    _target_: str = "paddle.optimizer.Lamb"
    learning_rate: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    lamb_weight_decay: float = 0


def register_optimizer_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="optimizer",
        name="adam",
        node=AdamConf,
    )
    cs.store(
        group="optimizer",
        name="sgd",
        node=SGDConf,
    )
    cs.store(
        group="optimizer",
        name="bfgs",
        node=BFGSConf,
    )
    cs.store(
        group="optimizer",
        name="adadelta",
        node=AdadeltaConf,
    )
    cs.store(
        group="optimizer",
        name="adagrad",
        node=AdagradConf,
    )
    cs.store(
        group="optimizer",
        name="adamw",
        node=AdamWConf,
    )
    cs.store(
        group="optimizer",
        name="adamax",
        node=AdamaxConf,
    )
    cs.store(
        group="optimizer",
        name="rmsprop",
        node=RMSpropConf,
    )
    cs.store(
        group="optimizer",
        name="lamb",
        node=LambConf,
    )
