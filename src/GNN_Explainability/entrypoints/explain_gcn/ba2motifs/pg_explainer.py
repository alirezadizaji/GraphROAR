import os.path as osp
import os

from dig.xgraph.method import PGExplainer
from dig.xgraph.models import *
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops

from GNN_Explainability.config.base_config import BaseConfig

from ....enums import Dataset
from ...core.main import MainEntrypoint
from ....utils.visualization import visualization


class Entrypoint(MainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=4,
            try_name='ba2motifs_pgexplainer',
            dataset_name=Dataset.BA2Motifs,
        )
        conf.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        conf.num_epochs = 10
        conf.save_log_in_file = False
        conf.shuffle_training = True
        conf.batch_size = 1

        conf.base_model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        conf.base_model.load_state_dict(torch.load('./GNN_Explainability/checkpoints/ba2motifs_gin_3l.pt', map_location=conf.device))

        super(Entrypoint, self).__init__(conf)

    def run(self):
        
        explainer = PGExplainer(self.conf.base_model, in_channels=600, 
                device=self.conf.device, explain_graph=True,
                epochs=self.conf.num_epochs, 
                lr=3e-3, coff_size=0.03, coff_ent=5e-4, t0=5.0, t1=1.0, sample_bias=0.0)

        dataset = []
        for i, data in enumerate(self.conf.train_loader):
            data: Data = data[0]
            data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            dataset.append(data)

        explainer.train_explanation_network(dataset)

        torch.save(explainer.state_dict(), './GNN_Explainability/checkpoints/ba2motifs_gin_3l_pgexplainer.pt')
        # explainer.load_state_dict(torch.load('./GNN_Explainability/checkpoints/ba2motifs_gin_3l_pgexplainer.pt', map_location=self.conf.device))

        for loader in [self.conf.train_loader, self.conf.val_loader, self.conf.test_loader]:
            for i, data in enumerate(loader):
                data: Data = data[0]
                if data.y.item() == 0:
                    continue
                data.edge_index = remove_self_loops(data.edge_index)[0]
                
                print(f'explain graph {i} gt label {data.y}', flush=True)
                _, masks, _ = \
                        explainer(data.x, data.edge_index, y=data.y)
                
                pos = visualization(data, data.y.item())
                edge_mask = masks[0]

                data.edge_index = data.edge_index[:, edge_mask >= 0.5]
                visualization(data, data.y.item(), pos)
                file_dir = os.path.join('GNN_Explainability', 'data', 'ba_2motifs', 'explanation', 'pgexplainer')
                os.makedirs(file_dir, exist_ok=True)
                
                file_name = f"{i}"
                np.save(os.path.join(file_dir, file_name), edge_mask.detach().cpu().numpy())                