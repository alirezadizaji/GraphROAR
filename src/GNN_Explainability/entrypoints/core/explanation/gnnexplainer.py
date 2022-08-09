from typing import TYPE_CHECKING
from . import InstanceX

import torch
from dig.xgraph.method.gnnexplainer import GNNExplainer

if TYPE_CHECKING:
    from torch_geometric.data import Data

class GNNExplainerEntrypoint(InstanceX[GNNExplainer]):
    def explain_instance(self, data: 'Data'):
        y_pred = self.model(data=data).argmax(-1)
        out_x = self.explainer(data.x, data.edge_index, sparsity=self.conf.sparsity, 
                    num_classes=self.conf.num_classes, target_label=y_pred, mask_features=self.conf.mask_features)
                
        return out_x
    
    def get_edge_mask(self, out_x, data: 'Data') -> torch.Tensor:
        edge_masks = out_x[0]
        assert len(edge_masks) == 1, "target label is passed and edge_masks len must be one"
        edge_mask = edge_masks[0].data.sigmoid()

        # edge_mask should be replaced to have an identical order with edge_index
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
    