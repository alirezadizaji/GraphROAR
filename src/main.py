from importlib import import_module
import random
from sys import argv
from typing import TYPE_CHECKING

import numpy as np
import torch

from GNN_Explainability.utils.decorators.stdout_stderr_setter import stdout_stderr_setter
if TYPE_CHECKING:
    from GNN_Explainability.entrypoints.core.main import MainEntrypoint


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(12345)
    script = import_module(f"GNN_Explainability.entrypoints.{argv[1]}")
    entrypoint: 'MainEntrypoint' = getattr(script, 'Entrypoint')()

    if entrypoint.conf.save_log_in_file:
        entrypoint.run = stdout_stderr_setter(entrypoint.conf.save_dir)(entrypoint.run)
    entrypoint.run()