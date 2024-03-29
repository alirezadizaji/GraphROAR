import os
from random import shuffle
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .base_dataset import BaseDataset
from ..enums import DataSpec, Dataset

if TYPE_CHECKING:
    from torch import Tensor


class GraphClsDataset(BaseDataset):
    indices: List[int] = None
    """ it records the graph indices to be same for all separations """

    dataset_name: Dataset = None

    raw_folder_name: str = 'raw'
    """ foldername where the raw text files are located """
    
    processed_name: str = 'processed'
    """ processed foldername to save the processed text files """

    def _read_raw_file(self) -> Tuple['Tensor', 'Tensor', 'Tensor', 'Tensor']:

        datasetname = self.dataset_name
        raw_folder_name = self.raw_folder_name

        base_path = os.path.join("..", "data", datasetname)
        
        edge_index = np.loadtxt(
                    os.path.join(base_path, raw_folder_name, f"{datasetname}_A.txt"), 
                    delimiter=',')
        # drop duplicate edges (Exists in REDDIT-BINARY and IMDB-BINARY)
        num1 = edge_index.shape[0]
        df = pd.DataFrame(data=edge_index)
        df = df.drop_duplicates()
        edge_index = df.to_numpy().T
        edge_index = torch.from_numpy(edge_index).long()
        num2 = edge_index.size(1)
        print(f"{num1} out of {num2} edges remained after edge duplicate removal.", flush=True)

        node_to_graph_ind = (torch.from_numpy(
                np.loadtxt(
                    os.path.join(base_path, raw_folder_name, f"{datasetname}_graph_indicator.txt"), 
                    delimiter=','))
                .squeeze()
                .long())
        
        y = (torch.from_numpy(
                np.loadtxt(
                    os.path.join(base_path, raw_folder_name, f"{datasetname}_graph_labels.txt"), 
                    delimiter=','))
                .squeeze()
                .long())

        node_attr_path = os.path.join(base_path, raw_folder_name, f"{datasetname}_node_attributes.txt") 
        if os.path.exists(node_attr_path):
            node_attrs = (torch.from_numpy(np.loadtxt(node_attr_path, delimiter=','))
                    .squeeze())
        else:
            node_attrs = None

        node_label_path = os.path.join(base_path, raw_folder_name, f"{datasetname}_node_labels.txt") 
        if os.path.exists(node_label_path):
            node_labels_in_once = (torch.from_numpy(np.loadtxt(node_label_path, delimiter=','))
                    .squeeze())

            # node labels must start from zero
            node_labels_in_once = node_labels_in_once - node_labels_in_once.min()
        else:
            # All nodes are unique and therefore have identical features
            node_labels_in_once = torch.zeros_like(node_to_graph_ind)

        processed_name = self.processed_name
        os.makedirs(os.path.join(base_path, processed_name), exist_ok=True)
        # save processed files
        if node_attrs is not None:
            torch.save(node_attrs, os.path.join(base_path, processed_name, 'node_attr.pt'))
        torch.save(edge_index, os.path.join(base_path, processed_name, 'edge_index.pt'))
        torch.save(node_to_graph_ind, os.path.join(base_path, processed_name, 'node_to_graph_ind.pt'))
        torch.save(node_labels_in_once, os.path.join(base_path, processed_name, 'node_labels_in_once.pt'))
        torch.save(y, os.path.join(base_path, processed_name, 'y.pt'))


    def _get_processed_files(self) -> Tuple['Tensor', 'Tensor', 'Tensor', 'Tensor']:
        r""" it returns four tensors:
        - edge_index
        - node_to_graph_ind: maps each node id to its graph id
        - node_labels_in_once: all unique nodes (in terms of their feature) exist
        - y: graph labels 
        """
        processed_name = self.processed_name
        base_path = os.path.join("..", "data", self.dataset_name)
        
        node_attr_path = os.path.join(base_path, processed_name, 'node_attr.pt')
        if os.path.exists(node_attr_path):
            node_attrs = torch.load(node_attr_path)
        else:
            node_attrs = None
        edge_index = torch.load(os.path.join(base_path, processed_name, 'edge_index.pt'))
        node_to_graph_ind = torch.load(os.path.join(base_path, processed_name, 'node_to_graph_ind.pt'))
        node_labels_in_once = torch.load(os.path.join(base_path, processed_name, 'node_labels_in_once.pt'))
        y = torch.load(os.path.join(base_path, processed_name, 'y.pt'))

        return edge_index, node_to_graph_ind, node_attrs, node_labels_in_once, y


    def __init__(self, dataspec: DataSpec):
        """
        Args:
            dataspec (DataSpec): data specification for which samples are loaded
        """
        self.dataspec: DataSpec = dataspec

        self.graphs: List[Data] = list()
        
        path = os.path.join("..", "data", self.dataset_name, self.processed_name)
        if not os.path.exists(path):
            self._read_raw_file()
        edge_index, node_to_graph_ind, node_attrs, node_labels_in_once, y = self._get_processed_files()
        node_labels_in_once = node_labels_in_once.long()
        
        # order labels
        unique_labels = torch.sort(torch.unique(y))[0]
        for i, label in enumerate(unique_labels.cpu().numpy()):
            y[y == label] = i

        num_unique_nodes = torch.unique(node_labels_in_once).numel()
        src = edge_index[0]

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
            nodes_mapping = torch.zeros(es.max().item() + 1).long()
            nodes_index = torch.unique(es)
            num_nodes = nodes_index.numel()
            new_nodes_index = torch.arange(num_nodes)
            nodes_mapping[nodes_index] = new_nodes_index
            es = nodes_mapping[es.flatten()].view(2, -1)

            # set attributes of nodes
            if node_attrs is not None:
                x = node_attrs[nodes_index]
            else:
                x = torch.zeros(num_nodes, num_unique_nodes)
                x[new_nodes_index, node_labels_in_once[nodes_index]] = 1
            x = x.float()

            graph = Data(x=x, edge_index=es, y=y[i])
            graph.name = str(i)
            self.graphs.append(graph)

        # determine indices of graphs and shuffle them
        if GraphClsDataset.indices is None:
            num_graphs = len(self.graphs)
            indices = np.arange(num_graphs).tolist()
            shuffle(indices)
            GraphClsDataset.indices = indices

        # reorder graphs and their labels
        self.graphs = [self.graphs[i] for i in GraphClsDataset.indices]
        y = y[GraphClsDataset.indices]

        train_inds, val_inds, test_inds = list(), list(), list()
        
        for label in torch.unique(y):
            label_inds = torch.nonzero(y == label)
            
            size = label_inds.numel()
            train_size = int(size * 0.8)
            test_size = int(size * 0.1)
            val_size = size - (train_size + test_size)

            train, test, val = torch.split(label_inds, [train_size, test_size, val_size])

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
