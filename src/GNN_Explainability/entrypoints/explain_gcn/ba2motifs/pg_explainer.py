import os.path as osp
import os

from dig.xgraph.method import PGExplainer
from dig.xgraph.models import *
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import DataLoader, Data

from ....data.ba_2motifs import BA2MotifsDataset
from ....enums.data_spec import DataSpec
from ...main import MainEntrypoint

class Entrypoint(MainEntrypoint):
    def run():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        train_loader = DataLoader(BA2MotifsDataset(DataSpec.TRAIN), shuffle=False)
        val_loader = DataLoader(BA2MotifsDataset(DataSpec.VAL), shuffle=False)
        test_loader = DataLoader(BA2MotifsDataset(DataSpec.TEST), shuffle=False)
        
        model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        model.to(device)
        model.load_state_dict(torch.load('./GNN_Explainability/checkpoints/ba2motifs_gin_3l.pt', map_location=device))
        
        explainer = PGExplainer(model, in_channels=600, device=device, explain_graph=True, epochs=100, k=3)

        for dataspec, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
            dataset = []
            for i, data in enumerate(loader):
                data: Data = data[0]
                dataset.append(data)

            explainer.train_explanation_network(dataset)

            for i, data in enumerate(loader):
                data: Data = data[0]
                print(f'explain graph {i} gt label {data.y}', flush=True)
                _, masks, _ = \
                        explainer(data.x, data.edge_index, y=data.y)
                
                edge_mask = masks[0].sigmoid()

                # Thanks to DIG GNNExplainer, edge_mask should be replaced to have an identical order with edge_index
                edge_mask_new = torch.full((data.edge_index.size(1),), fill_value=-100, dtype=torch.float)
                row, col = data.edge_index
                self_loop_mask = row == col    
                num = (~self_loop_mask).sum() 
                edge_mask_new[~self_loop_mask] = edge_mask[:num]

                if torch.any(self_loop_mask):
                    self_loop_node_inds = row[self_loop_mask]
                    edge_mask_new[self_loop_mask] = edge_mask[num:][self_loop_node_inds]

                if torch.any(edge_mask_new == -100):
                    raise Exception('there is an unhandled edge mask')
                
                edge_mask_new = edge_mask_new.cpu().numpy()
                file_dir = os.path.join('GNN_Explainability', 'data', 'ba2motifs', 'gnnexplainer', dataspec)
                os.makedirs(file_dir, exist_ok=True)
                
                file_name = f"{i}"
                np.save(os.path.join(file_dir, file_name), edge_mask_new)                