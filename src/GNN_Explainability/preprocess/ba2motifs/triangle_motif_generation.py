from typing import List, TYPE_CHECKING

import torch

from ...enums import *

if TYPE_CHECKING:
    from torch_geometric.data import Data


def create_new_graphs_with_triangles_motif(graphs_label1):
    graphs_label2: List['Data'] = list()

    for g in graphs_label1:
        # eliminate 23rd and 24th nodes
        g.x = g.x[:-2]

        row, col = g.edge_index
        mask = (row <= 22) & (col <= 22)
        g.edge_index = g.edge_index[:, mask]
        
        # triangle motif: connect 20th and 21st nodes
        new_edges = torch.tensor([[20, 22], [22, 20]])
        g.edge_index = torch.cat([g.edge_index, new_edges], dim=1)
        g.y = torch.tensor([2])
        graphs_label2.append(g)
    
    return graphs_label2

