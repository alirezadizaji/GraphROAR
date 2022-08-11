from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..enums.data_spec import DataSpec


class BA2MotifsDataset(Dataset):
    def __init__(self, dataspec: DataSpec):
        """
        Args:
            dataspec (DataSpec): data specification for which samples are loaded
        """
        self.dataspec: DataSpec = dataspec

        ba_2motifs = torch.load('../data/ba_2motifs/processed/data.pt')
        self.data: Data = ba_2motifs[0]
        self.data.name = [str(i) for i in range(self.data.y.numel())]
        
        train_inds, val_inds, test_inds = list(), list(), list()
        
        for label in torch.unique(self.data.y):
            label_inds = torch.nonzero(self.data.y == label)
            
            size = label_inds.numel()
            train_size = int(size * 0.8)
            val_size = int(size * 0.1)
            test_size = size - (train_size + val_size)

            train, val, test = torch.split(label_inds, [train_size, val_size, test_size])

            train_inds = train_inds + train.flatten().tolist()
            val_inds = val_inds + val.flatten().tolist()
            test_inds = test_inds + test.flatten().tolist()
            
        meta_info: Dict[str, torch.Tensor] = ba_2motifs[1]
        self.batch_nodes_inds: torch.Tensor = meta_info['x']
        self.batch_y_inds: torch.Tensor = meta_info['y']
        self.batch_edges_inds: torch.Tensor = meta_info['edge_index']

        if self.dataspec == DataSpec.TRAIN:
            self.graph_inds = train_inds
        elif self.dataspec == DataSpec.VAL:
            self.graph_inds = val_inds
        else:
            self.graph_inds = test_inds
    
    def __len__(self):
        return len(self.graph_inds)

    def __getitem__(self, inds) -> List[Data]:
        batch: List[Data] = []
        
        if isinstance(inds, int):
            inds = [inds]

        for idx in inds:
            graph_ind = self.graph_inds[idx]

            batch_node_ind_start = self.batch_nodes_inds[graph_ind]
            batch_node_ind_end = self.batch_nodes_inds[graph_ind + 1]
            x = self.data.x[batch_node_ind_start: batch_node_ind_end]

            batch_edge_ind_start = self.batch_edges_inds[graph_ind]
            batch_edge_ind_end = self.batch_edges_inds[graph_ind + 1]
            edge_index = self.data.edge_index[:, batch_edge_ind_start: batch_edge_ind_end]            

            y = self.data.y[self.batch_y_inds[graph_ind]]
            name = self.data.name[self.batch_y_inds[graph_ind]]
            data = Data(x=x, edge_index=edge_index, y=y)
            data.name = name
            batch.append(data)

        return batch