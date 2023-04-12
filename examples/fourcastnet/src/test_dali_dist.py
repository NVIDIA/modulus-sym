# TODO(akamenev): add proper license header.

from pathlib import Path
import tempfile
from termcolor import cprint

from modulus.sym.distributed import DistributedManager

from src.test_dali_dataset import (
    create_test_data,
    test_distributed_dali_loader,
)


if __name__ == "__main__":
    DistributedManager.initialize()
    m = DistributedManager()
    if not m.distributed:
        print(
            "Please run this test in distributed mode. For example, to run on 2 GPUs:\n\n"
            "mpirun -np 2 python ./src/test_dali_dist.py\n"
        )
        raise SystemExit(1)

    with tempfile.TemporaryDirectory("-data") as data_dir:
        data_path = create_test_data(Path(data_dir))

        test_distributed_dali_loader(data_path)

    cprint("Success!", "green")
