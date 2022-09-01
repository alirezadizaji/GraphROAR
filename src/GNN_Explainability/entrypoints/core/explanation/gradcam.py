from typing import TYPE_CHECKING

from . import InstanceX
from .utils import compatible_edge_mask

import torch
from dig.xgraph.method.gradcam import GradCAM

if TYPE_CHECKING:
    from torch_geometric.data import Data

class GradCAMEntrypoint(InstanceX[GradCAM]):
    def explain_instance(self, data: 'Data'):
        out_x = self.explainer(data.x, data.edge_index,
                    sparsity=self.conf.sparsity,
                    num_classes=self.conf.num_classes)
                
        return out_x
    
    def get_edge_mask(self, out_x, data: 'Data') -> torch.Tensor:
        edge_masks = out_x[0]
        y_pred = self.model(data=data).argmax(-1)
        edge_mask = edge_masks[y_pred.item()].data
                
        # edge_mask should be replaced to have an identical order with edge_index
        edge_mask_new = compatible_edge_mask(data, edge_mask)
        
        return edge_mask_new