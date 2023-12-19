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
# import pytest
# from modulus.sym.models.arch import FuncArch
# from modulus.sym.key import Key
# from modulus.sym.graph import Graph
# from modulus.sym.eq.pdes.navier_stokes import NavierStokes
# from modulus.sym.models.fully_connected import FullyConnectedArch
# from modulus.sym.manager import JitManager
# paddle.seed(seed=0)
# device = str('cuda:0' if paddle.device.cuda.device_count() >= 1 else 'cpu'
#     ).replace('cuda', 'gpu')
# >>>torch.backends.cuda.matmul.allow_tf32 = False


# @pytest.mark.parametrize('jit_activation', [True, False])
# def test_func_arch_graph_1(jit_activation):
#     """
#     Explicitly specify the needed derivative terms as Graph argument.
#     """
#     jit_manager = JitManager()
#     jit_manager.enabled = jit_activation
#     jit_manager.arch_mode = 'only_activation'
#     deriv_keys = [Key.from_str('u__x'), Key.from_str('u__x__x'), Key.
#         from_str('v__y'), Key.from_str('v__y__y')]
#     network = FullyConnectedArch(input_keys=[Key('x'), Key('y')],
#         output_keys=[Key('u'), Key('v')])
#     nodes = [network.make_node('ref_net', jit=False)]
#     ft_graph = Graph(nodes, [Key('x'), Key('y')], req_names=deriv_keys,
#         func_arch=True).to(device)
#     ref_graph = Graph(nodes, [Key('x'), Key('y')], req_names=deriv_keys,
#         func_arch=False).to(device)
#     if jit_activation:
#         assert isinstance(network._impl.layers[0].callable_activation_fn,
# >>>            torch.jit.ScriptFunction)
#     func_arch_node = None
#     for node in ft_graph.node_evaluation_order:
#         evaluate = node.evaluate
#         if isinstance(evaluate, FuncArch):
#             func_arch_node = node
#     assert func_arch_node is not None, 'No FuncArch found in the graph'
#     out_48 = paddle.rand(shape=[100, 1])
#     out_48.stop_gradient = not True
#     x = out_48
#     out_49 = paddle.rand(shape=[100, 1])
#     out_49.stop_gradient = not True
#     y = out_49
#     in_vars = {'x': x, 'y': y}
#     ft_out = ft_graph(in_vars)
#     ref_out = ref_graph(in_vars)
#     for k in ref_out.keys():
#         assert paddle.allclose(x=ref_out[k], y=ft_out[k], atol=5e-05).item()


# @pytest.mark.parametrize('func_arch_allow_partial_hessian', [True, False])
# def test_func_arch_graph_2(func_arch_allow_partial_hessian):
#     """
#     Test the graph could automatically add intermediate derivatives to
#     FuncArch.
#     """
#     flow_net = FullyConnectedArch(input_keys=[Key('x'), Key('y')],
#         output_keys=[Key('u'), Key('v'), Key('p')])
#     ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
#     nodes = ns.make_nodes() + [flow_net.make_node(name='flow_network', jit=
#         False)]
#     ft_graph = Graph(nodes, [Key('x'), Key('y')], req_names=Key.
#         convert_list(['continuity', 'momentum_x', 'momentum_y']), func_arch
#         =True, func_arch_allow_partial_hessian=func_arch_allow_partial_hessian
#         ).to(device)
#     ref_graph = Graph(nodes, [Key('x'), Key('y')], req_names=Key.
#         convert_list(['continuity', 'momentum_x', 'momentum_y']), func_arch
#         =False).to(device)
#     func_arch_node = None
#     for node in ft_graph.node_evaluation_order:
#         evaluate = node.evaluate
#         if isinstance(evaluate, FuncArch):
#             func_arch_node = node
#     assert func_arch_node is not None, 'No FuncArch found in the graph'
#     expected_outputs = ['u', 'v', 'p', 'u__y', 'v__x', 'u__x', 'v__y',
#         'u__x__x', 'v__y__y', 'u__y__y', 'v__x__x']
#     if not func_arch_allow_partial_hessian:
#         expected_outputs += ['p__y', 'p__x']
#     ft_outputs = [str(key) for key in func_arch_node.outputs]
#     assert len(ft_outputs) == len(expected_outputs)
#     assert sorted(ft_outputs) == sorted(expected_outputs)
#     out_50 = paddle.rand(shape=[100, 1])
#     out_50.stop_gradient = not True
#     x = out_50
#     out_51 = paddle.rand(shape=[100, 1])
#     out_51.stop_gradient = not True
#     y = out_51
#     in_vars = {'x': x, 'y': y}
#     ft_out = ft_graph(in_vars)
#     ref_out = ref_graph(in_vars)
#     for k in ref_out.keys():
#         assert paddle.allclose(x=ref_out[k], y=ft_out[k], atol=0.0001).item()


# def test_get_key_dim():
#     input_keys = [Key('x', 1), Key('y', 1), Key('z', 1)]
#     key_dims = FuncArch._get_key_dim(input_keys)
#     expected = {'x': 0, 'y': 1, 'z': 2}
#     for key in key_dims:
#         assert expected[key] == key_dims[key]
#     input_keys = [Key('x', 1), Key('y', 2), Key('z', 1)]
#     key_dims = FuncArch._get_key_dim(input_keys)
#     expected = {'x': 0, 'z': 3}
#     for key in key_dims:
#         assert expected[key] == key_dims[key]
#     input_keys = [Key('x', 100), Key('y', 1), Key('z', 1)]
#     key_dims = FuncArch._get_key_dim(input_keys)
#     expected = {'y': 100, 'z': 101}
#     for key in key_dims:
#         assert expected[key] == key_dims[key]


# if __name__ == '__main__':
#     test_func_arch_graph_1(True)
#     test_func_arch_graph_1(False)
#     test_func_arch_graph_2(True)
#     test_func_arch_graph_2(False)
#     test_get_key_dim()
