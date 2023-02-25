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

import logging
import enum
import inspect
import importlib
from omegaconf import OmegaConf
from inspect import signature

from modulus.sym import Key
from modulus.sym.models.layers import Activation
from modulus.sym.models.arch import Arch


logger = logging.getLogger(__name__)


class ModulusModels(object):
    _model_classes = {
        "afno": "AFNOArch",
        "distributed_afno": "DistributedAFNOArch",
        "deeponet": "DeepONetArch",
        "fno": "FNOArch",
        "fourier": "FourierNetArch",
        "fully_connected": "FullyConnectedArch",
        "conv_fully_connected": "ConvFullyConnectedArch",
        "fused_fully_connected": "FusedMLPArch",
        "fused_fourier": "FusedFourierNetArch",
        "fused_hash_encoding": "FusedGridEncodingNetArch",
        "hash_encoding": "MultiresolutionHashNetArch",
        "highway_fourier": "HighwayFourierNetArch",
        "modified_fourier": "ModifiedFourierNetArch",
        "multiplicative_fourier": "MultiplicativeFilterNetArch",
        "multiscale_fourier": "MultiscaleFourierNetArch",
        "pix2pix": "Pix2PixArch",
        "siren": "SirenArch",
        "super_res": "SRResNetArch",
    }
    # Dynamic imports (prevents dep warnings of unused models)
    _model_imports = {
        "afno": "modulus.sym.models.afno",
        "distributed_afno": "modulus.sym.models.afno.distributed",
        "deeponet": "modulus.sym.models.deeponet",
        "fno": "modulus.sym.models.fno",
        "fourier": "modulus.sym.models.fourier_net",
        "fully_connected": "modulus.sym.models.fully_connected",
        "conv_fully_connected": "modulus.sym.models.fully_connected",
        "fused_fully_connected": "modulus.sym.models.fused_mlp",
        "fused_fourier": "modulus.sym.models.fused_mlp",
        "fused_hash_encoding": "modulus.sym.models.fused_mlp",
        "hash_encoding": "modulus.sym.models.hash_encoding_net",
        "highway_fourier": "modulus.sym.models.highway_fourier_net",
        "modified_fourier": "modulus.sym.models.modified_fourier_net",
        "multiplicative_fourier": "modulus.sym.models.multiplicative_filter_net",
        "multiscale_fourier": "modulus.sym.models.multiscale_fourier_net",
        "pix2pix": "modulus.sym.models.pix2pix",
        "siren": "modulus.sym.models.siren",
        "super_res": "modulus.sym.models.super_res_net",
    }
    _registered_archs = {}

    def __new__(cls):
        obj = super(ModulusModels, cls).__new__(cls)
        obj._registered_archs = cls._registered_archs

        return obj

    def __contains__(self, k):
        return k.lower() in self._model_classes or k.lower() in self._registered_archs

    def __len__(self):
        return len(self._model_classes) + len(self._registered_archs)

    def __getitem__(self, k: str):
        assert isinstance(k, str), "Model type key should be a string"
        # Case invariant
        k = k.lower()
        # Import built-in archs
        if k in self._model_classes:
            return getattr(
                importlib.import_module(self._model_imports[k]), self._model_classes[k]
            )
        # Return user registered arch
        elif k in self._registered_archs:
            return self._registered_archs[k]
        else:
            raise ValueError("Invalid model type key")

    def keys(self):
        return list(self._model_classes.keys()) + list(self._registered_archs.keys())

    @classmethod
    def add_model(cls, key: str, value):
        key = key.lower()
        assert (
            not key in cls._model_classes
        ), f"Model type name {type_name} already registered! Must be unique."

        cls._registered_archs[key] = value


def register_arch(model: Arch, type_name: str):
    """Add a custom architecture to the Modulus model zoo

    Parameters
    ----------
    model : Arch
        Model class
    type_name : str
        Unique name to idenitfy model in the configs
    """
    ModulusModels.add_model(type_name, model)
