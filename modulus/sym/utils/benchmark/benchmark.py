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

import time
from typing import Any, Callable, Optional
import paddle
from paddle.profiler import RecordEvent, Profiler, ProfilerTarget


def timeit(
    func: Callable,
    *args,
    steps: int = 100,
    warmup: int = 10,
    run_profile: bool = False,
    verbose: bool = True,
    label: Optional[str] = None,
    label_padding: int = 35,
    cpu_timing: bool = False,
):
    """
    Returns time/step in ms.
    If run_profile is True, then return (time/step in ms, a captured cuda events table)
    """
    raise NotImplementedError("This function is not implemented yet.")

    if label is None:
        assert func.__name__, "please provide a label for this benchmark"
        label = func.__name__

    # warmup
    paddle.framework.core.nvprof_nvtx_push(f"{label}_warmup")
    for _ in range(warmup):
        func(*args)
    paddle.framework.core.nvprof_nvtx_pop()  # pop label_warmup

    # start timer
    if cpu_timing:
        paddle.device.cuda.synchronize()
        start = time.time()
    else:
        start_event = paddle.device.cuda.Event(enable_timing=True)
        start_event.record()

    paddle.framework.core.nvprof_nvtx_push(f"{label}")
    if run_profile:
        if verbose:
            print("\n" + "=" * 70 + " " + label + " " + "=" * 70)
        with Profiler(activities=[ProfilerTarget.GPU]) as prof:
            with RecordEvent("run_total"):
                # Here might not be equaivalent to the original record_function
                for i in range(steps):
                    paddle.framework.core.nvprof_nvtx_push(f"{i}th_iteration")
                    func(*args)
                    paddle.framework.core.nvprof_nvtx_pop()
        events = prof.key_averages()
        if verbose:
            print(
                events.table(
                    sort_by="self_cuda_time_total",
                    max_src_column_width=200,
                    row_limit=15,
                )
            )
    else:
        events = None
        for i in range(steps):
            paddle.framework.core.nvprof_nvtx_push(f"{i}th_iteration")
            func(*args)
            paddle.framework.core.nvprof_nvtx_pop()
    paddle.framework.core.nvprof_nvtx_pop()

    # stop timer
    if cpu_timing:
        paddle.device.cuda.synchronize()
        time_ms = ((time.time() - start) / steps) * 1000
    else:
        end_event = paddle.device.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        time_ms = start_event.elapsed_time(end_event) / steps

    if verbose:
        print(f"{label.ljust(label_padding)}: {time_ms:.3f} ms/step")

    if run_profile:
        return time_ms, events
    else:
        return time_ms


def profile(
    func: Callable,
    *args,
    **kwargs,
):
    """
    Simply a convenient wrapper of the timeit function with run_profile=True.

    Returns: (time/step in ms, a captured cuda events table)
    """
    return timeit(func, *args, run_profile=True, **kwargs)
