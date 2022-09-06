from importlib import import_module
import random
from sys import argv
from typing import TYPE_CHECKING

import numpy as np
import torch
from GNN_Explainability.context.constants import Constants
from GNN_Explainability.utils.decorators.stdout_stderr_setter import stdout_stderr_setter
if TYPE_CHECKING:
    from GNN_Explainability.entrypoints.core.main import MainEntrypoint


def global_seed(seed: int):
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
    # seeds for running
    # BA-2Motifs: 12345
    # MUTAG: 3423
    # REDDIT-BINARY: 12345

    Constants.GLOBAL_SEED = int(argv[1])
    
    global_seed(Constants.GLOBAL_SEED)
    script = import_module(f"GNN_Explainability.entrypoints.{argv[2]}")
    entrypoint: 'MainEntrypoint' = getattr(script, 'Entrypoint')()

    if entrypoint.conf.save_log_in_file:
        entrypoint.run = stdout_stderr_setter(entrypoint.conf.save_dir)(entrypoint.run)

    print(f"%%% RUNNING SEED {Constants.GLOBAL_SEED} %%%", flush=True)
    entrypoint.run()