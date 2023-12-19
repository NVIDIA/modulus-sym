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

import paddle
import modulus
from modulus.sym.key import Key
from modulus.sym.hydra import to_yaml, instantiate_arch
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.models.afno.distributed import DistributedAFNONet
from modulus.sym.distributed.manager import DistributedManager
import os

os.environ["MODEL_PARALLEL_SIZE"] = "2"


@modulus.sym.main(config_path="conf", config_name="config_AFNO")
def run(cfg: ModulusConfig) -> None:
    manager = DistributedManager()
    model_rank = manager.group_rank(name="model_parallel")
    model_size = manager.group_size(name="model_parallel")
    if not manager.cuda:
        print("WARNING: No GPUs available. Exiting...")
        return
    if manager.world_size % 2 != 0:
        print(
            "WARNING: Total world size not a multiple of model parallel size (2). Exiting..."
        )
        return
    input_keys = [Key("coeff", scale=(7.4836, 4.49996))]
    output_keys = [Key("sol", scale=(0.00574634, 0.00388433))]
    img_shape = 720, 1440
    model = instantiate_arch(
        input_keys=input_keys,
        output_keys=output_keys,
        cfg=cfg.arch.distributed_afno,
        img_shape=img_shape,
    )
    nodes = [model.make_node(name="Distributed AFNO", jit=cfg.jit)]
    model = model.to(manager.place)
    sample = {
        str(k): paddle.randn(shape=[1, k.size, *img_shape]).to(manager.place)
        for k in input_keys
    }
    for i in range(4):
        result = model(sample)
        loss = paddle.square(x=result["sol"]).sum()
        loss.backward()
    expected_result_shape = [1, output_keys[0].size, *img_shape]
    result_shape = list(result["sol"].shape)
    assert (
        result_shape == expected_result_shape
    ), f"Incorrect result size. Expected {expected_result_shape}, got {local_result_shape}"


if __name__ == "__main__":
    run()
