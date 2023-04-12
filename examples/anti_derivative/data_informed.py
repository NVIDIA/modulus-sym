import os
import sys
import warnings

import torch
import numpy as np

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig, to_yaml
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.models.deeponet import DeepONetArch
from modulus.sym.domain.constraint.continuous import DeepONetConstraint
from modulus.sym.domain.validator.discrete import GridValidator
from modulus.sym.dataset.discrete import DictGridDataset

from modulus.sym.key import Key


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # [init-model]
    trunk_net = instantiate_arch(
        cfg=cfg.arch.trunk,
        input_keys=[Key("x")],
        output_keys=[Key("trunk", 128)],
    )
    branch_net = instantiate_arch(
        cfg=cfg.arch.branch,
        input_keys=[Key("a", 100)],
        output_keys=[Key("branch", 128)],
    )
    deeponet = instantiate_arch(
        cfg=cfg.arch.deeponet,
        output_keys=[Key("u")],
        branch_net=branch_net,
        trunk_net=trunk_net,
    )

    nodes = [deeponet.make_node("deepo")]
    # [init-model]

    # [datasets]
    # load training data
    file_path = "data/anti_derivative.npy"
    if not os.path.exists(to_absolute_path(file_path)):
        warnings.warn(
            f"Directory {file_path} does not exist. Cannot continue. Please download the additional files from NGC <TODO: Add link here>"
        )
        sys.exit()

    data = np.load(to_absolute_path(file_path), allow_pickle=True).item()
    x_train = data["x_train"]
    a_train = data["a_train"]
    u_train = data["u_train"]

    # load test data
    x_test = data["x_test"]
    a_test = data["a_test"]
    u_test = data["u_test"]
    # [datasets]

    # [constraint]
    # make domain
    domain = Domain()

    data = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={"a": a_train, "x": x_train},
        outvar={"u": u_train},
        batch_size=cfg.batch_size.train,
    )
    domain.add_constraint(data, "data")
    # [constraint]

    # [validator]
    # add validators
    for k in range(10):
        invar_valid = {
            "a": a_test[k * 100 : (k + 1) * 100],
            "x": x_test[k * 100 : (k + 1) * 100],
        }
        outvar_valid = {"u": u_test[k * 100 : (k + 1) * 100]}
        dataset = DictGridDataset(invar_valid, outvar_valid)

        validator = GridValidator(nodes=nodes, dataset=dataset, plotter=None)
        domain.add_validator(validator, "validator_{}".format(k))
    # [validator]

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
