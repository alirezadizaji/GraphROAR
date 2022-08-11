from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..enums.data_spec import DataSpec


class MUTAGDataset(Dataset):
    def __init__(self, dataspec: DataSpec):
        """
        Args:
            dataspec (DataSpec): data specification for which samples are loaded
        """
        self.dataspec: DataSpec = dataspec

        self.graphs: List[Data] = list()

        edge_index = (torch.from_numpy(
            pd.read_csv("../data/MUTAG/raw/MUTAG_A.txt", sep=',', header=None).to_numpy())).t()
        
        node_to_graph_ind = (torch.from_numpy(
            pd.read_csv("../data/MUTAG/raw/MUTAG_graph_indicator.txt", header=None).to_numpy())).squeeze()
        src = edge_index[0]
        
        y = (torch.from_numpy(
            pd.read_csv("../data/MUTAG/raw/MUTAG_graph_labels.txt", header=None).to_numpy())).squeeze()
        
        # split edges into their graphs
        edge_to_graph_ind = node_to_graph_ind[src - 1]
        _, sorted_edge_inds = torch.sort(edge_to_graph_ind, stable=True)
        edge_index = edge_index[:, sorted_edge_inds]
        _, counts = torch.unique(edge_to_graph_ind, return_counts=True)
        split_edge_index = torch.split(edge_index, counts.tolist(), dim=1)
        
        assert y.numel() == len(split_edge_index), f"Number of graph labels ({y.numel()}) and edges splitted into graphs ({len(split_edge_index)}) does not match."
        
        for i, es in enumerate(split_edge_index):
            # node index should be started from zero
            es = es - 1

            # reset node indices to start from zero
            nodes_index = torch.zeros(es.max().item() + 1).long()
            values = torch.unique(es)
            num_nodes = values.numel()
            nodes_index[values] = torch.arange(num_nodes)
            es = nodes_index[es.flatten()].view(2, -1)

            label = (y[i] > 0).long()
            x = torch.ones(num_nodes, 10)
            graph = Data(x=x, edge_index=es, y=label)
            graph.name = str(i)
            self.graphs.append(graph)
        
        train_inds, val_inds, test_inds = list(), list(), list()
        
        for label in torch.unique(y):
            label_inds = torch.nonzero(y == label)
            
            size = label_inds.numel()
            train_size = int(size * 0.8)
            val_size = int(size * 0.1)
            test_size = size - (train_size + val_size)

            train, val, test = torch.split(label_inds, [train_size, val_size, test_size])

            train_inds = train_inds + train.flatten().tolist()
            val_inds = val_inds + val.flatten().tolist()
            test_inds = test_inds + test.flatten().tolist()
            
        if self.dataspec == DataSpec.TRAIN:
            self.graph_inds = train_inds
        elif self.dataspec == DataSpec.VAL:
            self.graph_inds = val_inds
        else:
            self.graph_inds = test_inds
    
    def __len__(self):
        return len(self.graph_inds)
    
    def __getitem__(self, inds) -> List[Data]:
        if isinstance(inds, int):
            inds = [inds]
        
        batch: List[Data] = list()
        for ix in inds:
            graph_ind = self.graph_inds[ix]
            batch.append(self.graphs[graph_ind])
        
        return batch
