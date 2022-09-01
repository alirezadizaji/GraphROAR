import os
from typing import TYPE_CHECKING, List

from dig.xgraph.method import PGExplainer
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops

from GNN_Explainability.entrypoints.core.explanation.utils.compatible_edge_mask import compatible_edge_mask

from ....enums import *
from ...core.explanation.core.main import ExplainerEntrypoint
from ....utils.visualization import visualization
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

    def run(self):
        conf: 'PGExplainerConfig' = self.conf
        if conf.explainer_load_dir is not None:
            self.explainer.load_state_dict(torch.load(conf.explainer_load_dir, map_location=self.conf.device))
        else:
            dataset: List[Data] = []

            for data in self.train_loader:
                data: Data = data[0]
                data.edge_index = add_remaining_self_loops(data.edge_index)[0]
                dataset.append(data)
            
            self.explainer.train_explanation_network(dataset)
            
            model_weight_path = os.sep.join([conf.save_dir, 'weights', 'explainer.pt'])
            torch.save(self.explainer.state_dict(), model_weight_path)

        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            for data in loader:
                data: Data = data[0]
                edge_index = add_remaining_self_loops(data.edge_index)
                
                print(f"graph {data.name[0]}, label {data.y.item()}", flush=True)
                _, masks, _ = self.explainer(data.x, edge_index, y=data.y)
                edge_mask = masks[0]
                
                edge_mask_new = compatible_edge_mask(data, edge_mask)
                self.visualize_sample(data, edge_mask)

                if conf.edge_mask_save_dir is not None:
                    os.makedirs(conf.edge_mask_save_dir, exist_ok=True)
                    file_name = f"{data.name[0]}"
                    np.save(os.path.join(conf.edge_mask_save_dir, file_name), edge_mask_new)                