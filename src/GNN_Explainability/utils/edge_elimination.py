from dataclasses import dataclass
import os
from platform import node
from typing import Callable, List, Tuple
from typing_extensions import Protocol

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from .symmetric_edge_mask import symmetric_edges


class EdgeEliminatorInitializer(Protocol):
    def __call__(self, root_dir: str, ratio: float, symmetric: bool) -> Callable[[Batch], Batch]:
        ...

@dataclass
class EdgeEliminatorArgs:
    root_dir: str
    ratio: float
    symmetric: bool = False
    eliminate_top_most: bool = True
    eliminate_node: bool = False

    @property
    def item(self):
        return dict(
            root_dir=self.root_dir,
            ratio=self.ratio,
            symmetric=self.symmetric,
            eliminate_top_most=self.eliminate_top_most,
            eliminate_node=self.eliminate_node
        )


def init_edge_eliminator(root_dir: str, ratio: float, symmetric: bool,
        eliminate_top_most: bool, eliminate_node: bool) -> Callable[[Batch], Batch]:
    
    def _edge_eliminator(data: Batch):
        graphs = data.to_data_list()

        for g in graphs:
            edge_index: torch.Tensor = g.edge_index

            edge_mask_dir = os.path.join(root_dir, f"{g.name}.npy")
            edge_mask = np.load(edge_mask_dir)

            edge_mask = edge_index.new_tensor(edge_mask, dtype=torch.float)
            if symmetric:
                edge_mask = symmetric_edges(edge_index, edge_mask)
        
            k = int(edge_mask.numel() * ratio)
            if symmetric and k % 2 == 1:
                if eliminate_top_most:
                    k = k - 1            
                else:
                    k = k + 1
            _, inds = torch.topk(edge_mask, k)
            
            if eliminate_top_most:
                mask = torch.ones_like(edge_mask).bool()
                mask[inds] = False
            else:
                mask = torch.zeros_like(edge_mask).bool()
                mask[inds] = True

            masked_edges = edge_index[:, mask] 
            
            # If true then eliminate nodes whose connected edges are totally eliminated.
            if eliminate_node:
                # If all edges are removed, then (just to enable training) instead of node eliminating, make all of them the same
                if masked_edges.numel() == 0:
                    g.x = torch.zeros_like(g.x)
                else:
                    B = g.x.size(0)
                    node_indices, _ = torch.sort(torch.unique(edge_index), descending=False)
                    indices_to_indices = {k: v for k, v in 
                            zip(node_indices.cpu().numpy(), range(B))}
                    node_mask = torch.zeros(B).bool()
                    remained: torch.Tensor = torch.unique(masked_edges)
                    real_indices = list(map(lambda k: indices_to_indices[k.item()], remained))
                    node_mask[real_indices] = True
                    g.x = g.x[node_mask]

            g.edge_index = masked_edges


        data: Batch = Batch.from_data_list(graphs)
        return data

    return _edge_eliminator

def edge_elimination_hook(arguments: EdgeEliminatorArgs, skip_during_eval: bool = False):
    edge_eliminator = init_edge_eliminator(**arguments.item)
    def _hook(model: nn.Module, inp: Tuple[List[Batch]]) -> Tuple[Batch]:
        if isinstance(inp[0], Batch):
            data: Batch = inp[0]
        else:
            data: Batch = inp[0][0]
        
        # If necessary skip during evaluation if model is not in training mode
        if skip_during_eval and not model.training:
            pass
        else:
            data = edge_eliminator(data) 
        inp = ([data])
        return inp

    return _hook