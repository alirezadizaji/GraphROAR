import os
from typing import List, Tuple
from typing_extensions import Protocol

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from .decorators.func_counter import counter

class ForwardPreHook(Protocol):
    def __call__(self, module: nn.Module, inp: Tuple[Batch]) -> Tuple[Batch]:
        ...

def edge_elimination(root_dir: str, ratio: float, thd:float=0.5) -> ForwardPreHook:
    
    def _hook(_: nn.Module, inp: Tuple[List[Batch]]) -> Tuple[Batch]:
        data: Batch = inp[0][0]
        graphs = data.to_data_list()

        for g in graphs:
            edge_index = g.edge_index

            edge_mask_dir = os.path.join(root_dir, f"{g.name}.npy")
            edge_mask = np.load(edge_mask_dir)

            edge_mask = edge_index.new_tensor(edge_mask, dtype=torch.float)
            k = int(edge_mask[edge_mask >= thd].numel() * (1 - ratio))
            _, inds = torch.topk(edge_mask, k)
            g.edge_index = edge_index[:, inds]

        data: Batch = Batch.from_data_list(graphs)
        inp = ([data])
        return inp
    
    return _hook