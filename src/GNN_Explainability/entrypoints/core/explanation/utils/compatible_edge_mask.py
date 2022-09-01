from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data

def compatible_edge_mask(data: 'Data', edge_mask: 'torch.Tensor') -> 'torch.Tensor':
    """ compatibles the provided edge mask with the corresponding edge_index of data;
    This is because most of explanation methods implemented by DIG use  'edge reordering'
    in order to put the self-loop edges at the end of array (checkout the `add_remaining_self_loops` method of torch_geometric).
    """

    edge_mask_new = torch.full((data.edge_index.size(1),), fill_value=-100, dtype=torch.float, device=edge_mask.device)
    row, col = data.edge_index
    self_loop_mask = row == col    
    num = (~self_loop_mask).sum() 
    
    edge_mask_new[~self_loop_mask] = edge_mask[:num]

    if torch.any(self_loop_mask):
        self_loop_node_inds = row[self_loop_mask]
        edge_mask_new[self_loop_mask] = edge_mask[num:][self_loop_node_inds]

    if torch.any(edge_mask_new == -100):
        raise Exception('there is an unhandled edge mask')
    
    return edge_mask_new
