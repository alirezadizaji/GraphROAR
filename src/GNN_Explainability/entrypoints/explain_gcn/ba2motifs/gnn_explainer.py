import argparse
import random
import os.path as osp
import os

from dig.xgraph.method import GNNExplainer
from dig.xgraph.models import *
import numpy as np
import torch
from torch_geometric.data import DataLoader

from dig.xgraph.models import *
import numpy as np
import torch
from torch_geometric.data import DataLoader, Data
from torch.optim import Adam

from ....config.base_config import BaseConfig
from ....dataset.ba_2motifs import BA2MotifsDataset
from ....enums import *
from ...main import MainEntrypoint
from ....utils.visualization import visualization


class Entrypoint(MainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=2,
            try_name='ba2motifs_gnnexplainer',
            dataset_name=Dataset.BA2Motifs,
        )
        conf.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        conf.num_epochs = 500
        conf.save_log_in_file = False
        conf.shuffle_training = False
        
        conf.base_model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        conf.base_model.to(conf.device)
        conf.base_model.load_state_dict(torch.load('./GNN_Explainability/checkpoints/ba2motifs_gin_3l.pt', map_location=conf.device))

        super(Entrypoint, self).__init__(conf)
        
    def run(self):
        explainer = GNNExplainer(self.conf.base_model, epochs=self.conf.num_epochs, lr=0.01, explain_graph=True)

        for dataspec, loader in zip(
                ['train', 'val', 'test'], 
                [self.conf.train_loader, self.conf.val_loader, self.conf.test_loader]):
            for i, data in enumerate(loader):
                data: Data = data[0]

                print(f'explain graph {i} gt label {data.y}', flush=True)
                data.to(self.conf.device)

                edge_masks, _, _ = \
                    explainer(data.x, data.edge_index, sparsity=0.0, num_classes=2, target_label=data.y, mask_features=True)

                edge_mask = edge_masks[data.y.item()].data.sigmoid()

                # Thanks to DIG GNNExplainer, edge_mask should be replaced to have an identical order with edge_index
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
                
                file_dir = os.path.join('..', 'data', 'ba_2motifs', 'explanation', 'gnnexplainer', dataspec)
                os.makedirs(file_dir, exist_ok=True)
                file_name = f"{i}"
                edge_mask_new = edge_mask_new.cpu().numpy()
                np.save(os.path.join(file_dir, file_name), edge_mask_new)                
