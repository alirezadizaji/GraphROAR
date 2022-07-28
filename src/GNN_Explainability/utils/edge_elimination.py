import os
from typing import Tuple
from typing_extensions import Protocol

import numpy as np
import torch
from torch import nn

from .decorators.func_counter import counter

class ForwardPreHook(Protocol):
    def __call__(self, module: nn.Module, inp: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        ...

def edge_elimination(root_dir: str, data_spec: str, percentage: float,
        thd:float=0.5) -> ForwardPreHook:
    
    @counter(0)
    def _hook(module: nn.Module, inp: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        inp = list(inp)

        edge_mask_dir = os.path.join(root_dir, data_spec, f"{_hook.call}.npz")
        edge_mask = (np.load(edge_mask_dir) >= thd).astype(np.uint8)
        
        for inpi in inp:
            if inpi.size(0) == 2:
                break
        edge_index = inp[inpi]
        edge_mask = edge_index.new_tensor(edge_mask)
        
        k = int(torch.nonzero(edge_mask).numel() * percentage)
        _, inds = torch.topk(edge_mask, k)
        edge_index = edge_index[inds]
        
        inp[inpi] = edge_index
        return tuple(inp)
        