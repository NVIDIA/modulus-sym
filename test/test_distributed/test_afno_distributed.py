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
from modulus.sym.hydra import to_yaml, instantiate_arch
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.models.afno.distributed import DistributedAFNONet
from modulus.sym.distributed.manager import DistributedManager
import os

os.environ["MODEL_PARALLEL_SIZE"] = "2"


@modulus.sym.main(config_path="conf", config_name="config_AFNO")
def run(cfg: ModulusConfig) -> None:
    input_is_matmul_parallel = False
    output_is_matmul_parallel = False
    in_chans = 3
    out_chans = 10
    embed_dim = 768
    manager = DistributedManager()
    if not manager.cuda:
        print("WARNING: No GPUs available. Exiting...")
        return
    if manager.world_size % 2 != 0:
        print(
            "WARNING: Total world size not a multiple of model parallel size (2). Exiting..."
        )
        return
    model = DistributedAFNONet(
        img_size=(720, 1440),
        patch_size=(4, 4),
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        input_is_matmul_parallel=input_is_matmul_parallel,
        output_is_matmul_parallel=output_is_matmul_parallel,
    ).to(manager.place)
    model_rank = manager.group_rank(name="model_parallel")
    model_size = manager.group_size(name="model_parallel")
    expected_embed_dim_local = embed_dim // model_size
    assert (
        model.embed_dim_local == expected_embed_dim_local
    ), f"Incorrect local embedding size. Expected {expected_embed_dim_local}, got {model.embed_dim_local}"
    sample = paddle.randn(shape=[1, in_chans, 720, 1440])
    local_in_chans_start = 0
    local_in_chans_end = in_chans
    if input_is_matmul_parallel:
        chunk = (in_chans + model_size - 1) // model_size
        local_in_chans_start = model_rank * chunk
        local_in_chans_end = min(in_chans, local_in_chans_start + chunk)
    local_sample = sample[:, local_in_chans_start:local_in_chans_end, :, :].to(
        manager.place
    )
    for i in range(4):
        local_result = model(local_sample)
        loss = paddle.square(x=local_result).sum()
        loss.backward()
    local_out_chans = out_chans
    if output_is_matmul_parallel:
        chunk = (out_chans + model_size - 1) // model_size
        local_out_chans_start = model_rank * chunk
        local_out_chans_end = min(out_chans, local_out_chans_start + chunk)
        local_out_chans = local_out_chans_end - local_out_chans_start
    expected_result_shape = [1, local_out_chans, 720, 1440]
    local_result_shape = list(local_result.shape)
    assert (
        local_result_shape == expected_result_shape
    ), f"Incorrect result size. Expected {expected_result_shape}, got {local_result_shape}"


if __name__ == "__main__":
    run()
