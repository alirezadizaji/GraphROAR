from dataclasses import dataclass
import os
from typing import Callable, List, Tuple
from typing_extensions import Protocol

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from .decorators.func_counter import counter

class EdgeEliminatorInitializer(Protocol):
    def __call__(self, root_dir: str, ratio: float, symmetric: bool) -> Callable[[Batch], Batch]:
        ...

@dataclass
class Arguments:
    root_dir: str
    ratio: float
    symmetric: bool = False

def init_edge_eliminator(root_dir: str, ratio: float, symmetric: bool) -> Callable[[Batch], Batch]:
    def _edge_eliminator(data: Batch):
        graphs = data.to_data_list()

        for g in graphs:
            edge_index: torch.Tensor = g.edge_index

            edge_mask_dir = os.path.join(root_dir, f"{g.name}.npy")
            edge_mask = np.load(edge_mask_dir)

            edge_mask = edge_index.new_tensor(edge_mask, dtype=torch.float)
            if symmetric:
                num_nodes = edge_index.unique().numel()
                edge_mask_asym = torch.sparse_coo_tensor(edge_index, 
                        edge_mask, (num_nodes, num_nodes)).to_dense()
                edge_mask_sym = (edge_mask_asym + edge_mask_asym.T) / 2
                edge_mask = edge_mask_sym[edge_index[0], edge_index[1]]
        
            k = int(edge_mask.numel() * ratio)
            _, inds = torch.topk(edge_mask, k)
            mask = torch.ones_like(edge_mask).bool()
            mask[inds] = False
            g.edge_index = edge_index[:, mask]

        data: Batch = Batch.from_data_list(graphs)
        return data
        
    return _edge_eliminator

def edge_elimination_hook(arguments: Arguments):
    edge_eliminator = init_edge_eliminator(**arguments)
    def _hook(_: nn.Module, inp: Tuple[List[Batch]]) -> Tuple[Batch]:
        if isinstance(inp[0], Batch):
            data: Batch = inp[0]
        else:
            data: Batch = inp[0][0]
        data = edge_eliminator(data) 
        inp = ([data])
        return inp

    return _hook