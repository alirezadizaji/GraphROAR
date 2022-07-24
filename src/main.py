from importlib import import_module
import random
from sys import argv

import numpy as np
import torch

from GNN_Explainability.entrypoints.main import Entrypoint

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
    script = import_module(f"entrypoints.{argv[1]}")
    entrypoint: Entrypoint = getattr(script, 'Entrypoint')
    entrypoint.run()