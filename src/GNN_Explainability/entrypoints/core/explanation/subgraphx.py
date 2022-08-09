from typing import TYPE_CHECKING
from . import InstanceX

import torch
from dig.xgraph.method.subgraphx import SubgraphX, find_closest_node_result

if TYPE_CHECKING:
    from torch_geometric.data import Data

class SubgraphXEntrypoint(InstanceX[SubgraphX]):
    def explain_instance(self, data: 'Data'):
        y_pred = self.model(data=data).argmax(-1).item()
        out_x = self.explainer.explain(data.x, data.edge_index,
                    max_nodes=self.conf.get_max_nodes(data),
                    label=y_pred)
        return out_x
    
    def get_edge_mask(self, out_x, data: 'Data') -> torch.Tensor:
        explain_result = out_x[0]
        explain_result = self.explainer.read_from_MCTSInfo_list(explain_result)
        explanation = find_closest_node_result(explain_result, max_nodes=self.conf.get_max_nodes(data))
        edge_mask = data.edge_index[0].cpu().apply_(lambda x: x in explanation.coalition).bool() & \
                    data.edge_index[1].cpu().apply_(lambda x: x in explanation.coalition).bool()
        edge_mask = edge_mask.float()
        return edge_mask
    