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

""" Monitor for Solver class
"""

import numpy as np

from modulus.sym.domain.monitor import Monitor
from modulus.sym.domain.constraint import Constraint
from modulus.sym.graph import Graph
from modulus.sym.key import Key
from modulus.sym.constants import TF_SUMMARY
from modulus.sym.distributed import DistributedManager
from modulus.sym.utils.io import dict_to_csv, csv_to_dict


class PointwiseMonitor(Monitor):
    """
    Pointwise Inferencer that allows inferencing on pointwise data

    Parameters
    ----------
    invar : Dict[str, np.ndarray (N, 1)]
        Dictionary of numpy arrays as input.
    output_names : List[str]
        List of outputs needed for metric.
    metrics : Dict[str, Callable]
        Dictionary of pytorch functions whose input is a dictionary
        torch tensors whose keys are the `output_names`. The keys
        to `metrics` will be used to label the metrics in tensorboard/csv outputs.
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.
    requires_grad : bool = False
        If automatic differentiation is needed for computing results.
    """

    def __init__(self, invar, output_names, metrics, nodes, requires_grad=False):

        # construct model from nodes
        self.requires_grad = requires_grad
        self.model = Graph(
            nodes, Key.convert_list(invar.keys()), Key.convert_list(output_names)
        )
        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)

        # set metrics
        self.metrics = metrics
        self.monitor_outvar_store = {}

        # set invar
        self.invar = Constraint._set_device(invar, device=self.device)

    def save_results(self, name, writer, step, data_dir):

        # run forward inference
        invar = Constraint._set_device(
            self.invar, device=self.device, requires_grad=self.requires_grad
        )
        outvar = self.model(invar)
        metrics = {key: func({**invar, **outvar}) for key, func in self.metrics.items()}

        for k, m in metrics.items():
            # add tensorboard scalars
            if TF_SUMMARY:
                writer.add_scalar("monitor/" + name + "/" + k, m, step, new_style=True)
            else:
                writer.add_scalar("Monitors/" + name + "/" + k, m, step, new_style=True)

            # write csv files
            if k not in self.monitor_outvar_store.keys():
                try:
                    self.monitor_outvar_store[k] = csv_to_dict(data_dir + k + ".csv")
                except:
                    self.monitor_outvar_store[k] = {
                        "step": np.array([[step]]),
                        k: m.detach().cpu().numpy().reshape(-1, 1),
                    }
            else:
                monitor_outvar = {
                    "step": np.array([[step]]),
                    k: m.detach().cpu().numpy().reshape(-1, 1),
                }
                self.monitor_outvar_store[k] = {
                    key: np.concatenate([value_1, value_2], axis=0)
                    for (key, value_1), (key, value_2) in zip(
                        self.monitor_outvar_store[k].items(), monitor_outvar.items()
                    )
                }
            dict_to_csv(self.monitor_outvar_store[k], filename=data_dir + k + ".csv")
        return metrics
