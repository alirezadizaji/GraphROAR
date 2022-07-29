from importlib import import_module
import os
import random
from sys import argv

import numpy as np
import torch

from GNN_Explainability.entrypoints.main import MainEntrypoint
from GNN_Explainability.utils.decorators.stdout_stderr_setter import stdout_stderr_setter

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
    entrypoint: MainEntrypoint = getattr(script, 'Entrypoint')()

    if entrypoint.conf.save_log_in_file:
        save_dir = os.path.join('..', 'results', f"{entrypoint.conf.try_num}_{entrypoint.conf.try_name}")
        stdout_stderr_setter(save_dir)(entrypoint.run)()
    else:
        entrypoint.run()