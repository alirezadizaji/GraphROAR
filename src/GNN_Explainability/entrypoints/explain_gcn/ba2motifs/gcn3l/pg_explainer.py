import os.path as osp
import os

from dig.xgraph.method import PGExplainer
from dig.xgraph.models import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops

from .....config.base_config import BaseConfig, TrainingConfig
from .....enums import *
from ....core.main import MainEntrypoint
from .....utils.visualization import visualization


class Entrypoint(MainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=39,
            try_name='pgexplainer_gcn3l',
            dataset_name=Dataset.BA2Motifs,
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=False,
            training_config=TrainingConfig(30, OptimType.ADAM, batch_size=1),
        )
        conf.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        model = GCN_3l_BN(model_level='graph', dim_node=10, dim_hidden=20, num_classes=2)
        model.load_state_dict(torch.load('../results/1_gcn3l_BA2Motifs/weights/16', map_location=conf.device))

        super(Entrypoint, self).__init__(conf, model)

    def run(self):
        
        explainer = PGExplainer(self.model, in_channels=40, 
                device=self.conf.device, explain_graph=True,
                epochs=self.conf.training_config.num_epochs, 
                lr=3e-3, coff_size=0.0, coff_ent=0.0, t0=5.0, t1=1.0, sample_bias=0.0)

        dataset = []
        for i, data in enumerate(self.train_loader):
            data: Data = data[0]
            data.edge_index = remove_self_loops(data.edge_index)[0]
            dataset.append(data)

        # explainer.train_explanation_network(dataset)


        # torch.save(explainer.state_dict(), '../results/39_pgexplainer_gcn3l_BA2Motifs/weights/model.pt')
        explainer.load_state_dict(torch.load('../results/39_pgexplainer_gcn3l_BA2Motifs/weights/model.pt', map_location=self.conf.device))

        i_total = 0
        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            for i, data in enumerate(loader):
                i_total += 1
                data: Data = data[0]
                
                edge_index = data.edge_index
                mask = edge_index[0] != edge_index[1]
                mask = mask.cpu().numpy()
                
                print(f'explain graph {i} gt label {data.y}', flush=True)
                _, masks, _ = \
                        explainer(data.x, edge_index[:, mask], y=data.y)
                
                edge_mask = masks[0]

                if i_total <= 20:
                    pos = visualization(data, f"{data.name[0]}_{data.y.item()}_org", 
                    save_dir='../results/39_pgexplainer_gcn3l_BA2Motifs/visualizations')
                    plt.close()

                    
                    data.edge_index = data.edge_index[:, edge_mask.topk(10)[1]]
                    visualization(data, f"{data.name[0]}_{data.y.item()}_X", pos, 
                    save_dir='../results/39_pgexplainer_gcn3l_BA2Motifs/visualizations')
                    plt.close()
                
                new_edge_mask = np.ones_like(mask, dtype=np.float)
                new_edge_mask[mask] = edge_mask.detach().cpu().numpy()
                
                file_dir = os.path.join('..', 'data', 'ba_2motifs', 'explanation', 'gcn3l', 'pgexplainer')
                os.makedirs(file_dir, exist_ok=True)
                
                file_name = f"{data.name[0]}"
                np.save(os.path.join(file_dir, file_name), new_edge_mask)                