import os
from typing import TYPE_CHECKING, List

from dig.xgraph.method import PGExplainer
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops


from ....enums import *
from ...core.explanation.core.main import ExplainerEntrypoint
from .utils import compatible_edge_mask

if TYPE_CHECKING:
    from ....config import PGExplainerConfig
    from dig.xgraph.models import GNNBasic


class PGExplainerEntrypoint(ExplainerEntrypoint):
    def __init__(self, conf: 'PGExplainerConfig', model: 'GNNBasic', explainer: 'PGExplainer'):
        if conf.training_config.batch_size > 1:
            print("** WARNING: for instance-based explainer, batch size is changed to one", flush=True)
        conf.training_config.batch_size = 1
        
        super().__init__(conf, model)        

        self.explainer = explainer

    def get_edge_mask(self, out_x, data: 'Data') -> torch.Tensor:
        masks = out_x[1]
        edge_mask = masks[0]
        
        return edge_mask

    def _select_explainable_edges(self, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        conf: 'PGExplainerConfig' = self.conf
        edge_index = edge_index[:, edge_mask.topk(conf.topk_selection)[1]]

        return edge_index

    def run(self):
        conf: 'PGExplainerConfig' = self.conf
        if conf.explainer_load_dir is not None:
            self.explainer.load_state_dict(torch.load(conf.explainer_load_dir, map_location=self.conf.device))
        else:
            dataset: List[Data] = []

            for data in self.train_loader:
                data: Data = data[0]
                data.edge_index = remove_self_loops(data.edge_index)[0]
                dataset.append(data)
            
            self.explainer.train_explanation_network(dataset)
            
            model_weight_path = os.sep.join([conf.save_dir, 'weights', 'model.pt'])
            os.makedirs(os.path.dirname(model_weight_path), exist_ok=True)
            torch.save(self.explainer.state_dict(), model_weight_path)

        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            for data in loader:
                data: Data = data[0]
                row, col = data.edge_index
                mask = row != col
                edge_index = data.edge_index[:, mask]
                
                print(f"graph {data.name[0]}, label {data.y.item()}", flush=True)
                out_x = self.explainer(data.x, edge_index, y=data.y)
                edge_mask = self.get_edge_mask(out_x, data)
                
                edge_mask_new = torch.full((data.edge_index.size(1), ), fill_value=-torch.inf, dtype=torch.float32)
                edge_mask_new[mask] = edge_mask
                self.visualize_sample(data, edge_mask_new)

                edge_mask_new[~mask] = 1.0
                if conf.edge_mask_save_dir is not None:
                    os.makedirs(conf.edge_mask_save_dir, exist_ok=True)
                    file_name = f"{data.name[0]}"
                    np.save(os.path.join(conf.edge_mask_save_dir, file_name), edge_mask_new.detach().cpu().numpy())                