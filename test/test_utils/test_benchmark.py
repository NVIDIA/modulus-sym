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
import pytest
from modulus.sym.utils.benchmark import timeit

skip_if_no_gpu = pytest.mark.skipif(
    not paddle.device.cuda.device_count() >= 1,
    reason="There is no GPU to run this test",
)


@skip_if_no_gpu
def test_timeit():
    def func():
        paddle.zeros(shape=2**20).exp().cos().sin()

    cpu_timing_ms = timeit(func, cpu_timing=False)
    cuda_event_timing_ms = timeit(func, cpu_timing=True)
    assert cpu_timing_ms - cuda_event_timing_ms < 0.1


if __name__ == "__main__":
    test_timeit()
