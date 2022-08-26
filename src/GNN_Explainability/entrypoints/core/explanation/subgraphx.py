from typing import TYPE_CHECKING
from . import InstanceX

import torch
from dig.xgraph.method.subgraphx import SubgraphX, find_closest_node_result

if TYPE_CHECKING:
    from torch_geometric.data import Data

class SubgraphXEntrypoint(InstanceX[SubgraphX]):
    def explain_instance(self, data: 'Data'):
        # explainer is initiated here because `min_atoms` attribute might be filled based on the given instance number of edges
        self.explainer = SubgraphX(self.model, num_classes=self.conf.num_classes, device=self.conf.device, 
                        explain_graph=self.conf.explain_graph, reward_method=self.conf.reward_method,
                        min_atoms=self.conf.get_max_nodes(data), high2low=True, 
                        subgraph_building_method='split', verbose=True, rollout=10)
        
        y_pred = self.model(data=data).argmax(-1).item()
        out_x = self.explainer.explain(data.x, data.edge_index,
                    max_nodes=self.conf.get_max_nodes(data),
                    label=y_pred)
        
        return out_x
    
    def get_edge_mask(self, out_x, data: 'Data') -> torch.Tensor:
        explain_result = out_x[0]
        explain_result = self.explainer.read_from_MCTSInfo_list(explain_result)
        explanation = find_closest_node_result(explain_result, max_nodes=self.conf.get_max_nodes(data))
        edge_index = data.edge_index.clone()
        edge_mask = data.edge_index[0].cpu().apply_(lambda x: x in explanation.coalition).bool() & \
                    data.edge_index[1].cpu().apply_(lambda x: x in explanation.coalition).bool()
        
        data.edge_index = edge_index

        return edge_mask
    