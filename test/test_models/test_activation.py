# # Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import paddle
# import warnings
# import pytest
# from packaging import version
# from modulus.sym.manager import JitManager
# from modulus.sym.utils.benchmark import profile, timeit
# from modulus.sym.models.layers.activation import Activation, get_activation_fn
# skip_if_no_gpu = pytest.mark.skipif(not paddle.device.cuda.device_count() >=
#     1, reason='There is no GPU to run this test')


# def test_activation_jit():
#     jit_manager = JitManager()
#     jit_manager.enabled = True
#     jit_manager.arch_mode = 'only_activation'
#     for act in Activation:
#         act_scripted = get_activation_fn(act)
# >>>        assert isinstance(act_scripted, (torch.jit.ScriptFunction, torch.
#             jit.ScriptModule))

#     def sin(x):
#         return paddle.sin(x=x)
#     sin_scripted = get_activation_fn(sin)
# >>>    assert isinstance(sin_scripted, torch.jit.ScriptFunction)


# @skip_if_no_gpu
# def test_activation_fused_silu():
#     """
#     Make sure SiLU derivative kernels are fused when jit_manager.arch_mode == "only_activation".
#     We need to rely on the fused SiLU derivative kernels for AMP, because the unfused path
#     may have intermediate results that overflow the FP16 dynamic range.
#     """
#     jit_manager = JitManager()
#     jit_manager.enabled = True
#     jit_manager.arch_mode = 'only_activation'
#     jit_manager.use_nvfuser = True
#     silu_scripted = get_activation_fn(Activation.SILU)
# >>>    assert isinstance(silu_scripted, torch.jit.ScriptFunction)
#     device = 'cuda'
#     batch_size = 10000
#     out_47 = paddle.rand(shape=[batch_size, 512])
#     out_47.stop_gradient = not True
#     x = out_47
#     I_N = paddle.ones_like(x=x)

#     def run(func, order, *args):
#         paddle.framework.core.nvprof_nvtx_push('forward')
#         y = func(*args)
#         paddle.framework.core.nvprof_nvtx_pop()
#         if order >= 1:
#             paddle.framework.core.nvprof_nvtx_push('1st order')
#             y__x, = paddle.grad(outputs=y, inputs=[x], grad_outputs=I_N,
#                 create_graph=True)
#             paddle.framework.core.nvprof_nvtx_pop()
#         if order >= 2:
#             paddle.framework.core.nvprof_nvtx_push('2nd order')
#             y__x__x, = paddle.grad(outputs=y__x, inputs=[x], grad_outputs=
#                 I_N, create_graph=True)
#             paddle.framework.core.nvprof_nvtx_pop()
#         if order >= 3:
#             paddle.framework.core.nvprof_nvtx_push('3rd order')
#             y__x__x__x, = paddle.grad(outputs=y__x__x, inputs=[x],
#                 grad_outputs=I_N, create_graph=True)
#             paddle.framework.core.nvprof_nvtx_pop()

#     def cleanup_events(event_keys):
#         keys = ['cuLaunchKernel', 'cudaLaunchKernel', 'cudaDeviceSynchronize']
#         for evt in keys:
#             if evt in event_keys:
#                 event_keys.remove(evt)
#         return event_keys
#     silu = paddle.nn.functional.silu
#     timeit(run, silu, 1, x, label='silu_1st', verbose=True)
#     timeit(run, silu_scripted, 1, x, label='silu_scripted_1st', verbose=True)
#     timeit(run, silu, 2, x, label='silu_2nd', verbose=True)
#     timeit(run, silu_scripted, 2, x, label='silu_scripted_2nd', verbose=True)
#     timeit(run, silu, 3, x, label='silu_3rd', verbose=True)
#     timeit(run, silu_scripted, 3, x, label='silu_scripted_3rd', verbose=True)
#     verbose = False
#     _, events = profile(run, silu_scripted, 1, x, label='silu_scripted_1st',
#         verbose=verbose)
#     event_keys = cleanup_events([evt.key for evt in events])
#     num_kernels = len(event_keys)
#     print('silu_scripted_1st num_events: ', num_kernels)
#     if version.parse(paddle.__version__) >= version.parse('1.12.9'):
#         assert num_kernels == 2
#     else:
#         warnings.warn(
#             f'Fused SiLU is not supported for torch {paddle.__version__}')
#     _, events = profile(run, silu_scripted, 2, x, label='silu_scripted_2nd',
#         verbose=verbose)
#     event_keys = cleanup_events([evt.key for evt in events])
#     num_kernels = len(event_keys)
#     print('silu_scripted_2nd num_events: ', num_kernels)
#     if version.parse(paddle.__version__) >= version.parse('1.12.9'):
#         assert num_kernels == 3
#     else:
#         warnings.warn(
#             f'Fused SiLU is not supported for torch {paddle.__version__}')
#     _, events = profile(run, silu_scripted, 3, x, label='silu_scripted_3rd',
#         verbose=verbose)
#     event_keys = cleanup_events([evt.key for evt in events])
#     num_kernels = len(event_keys)
#     print('silu_scripted_3rd num_events: ', num_kernels)
#     if version.parse(paddle.__version__) >= version.parse('1.12.9'):
#         assert num_kernels <= 6
#     else:
#         warnings.warn(
#             f'Fused SiLU is not supported for torch {paddle.__version__}')


# if __name__ == '__main__':
#     test_activation_jit()
#     test_activation_fused_silu()
