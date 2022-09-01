from typing import TYPE_CHECKING

from . import InstanceX
from .utils import compatible_edge_mask

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
        edge_mask_new = compatible_edge_mask(data, edge_mask)
        
        return edge_mask_new
    