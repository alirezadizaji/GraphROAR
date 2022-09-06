from random import shuffle
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from .base_dataset import GraphClsDataset
from ..enums.data_spec import DataSpec

if TYPE_CHECKING:
    from torch import Tensor

class MUTAGDataset(GraphClsDataset):
    use_latest_version: bool = True

    def _read_file(self) -> Tuple['Tensor', 'Tensor', 'Tensor', 'Tensor']:
        if self.use_latest_version:
            foldername = 'raw'
        else:
            foldername = 'raw_org'
        edge_index = (torch.from_numpy(np.loadtxt(f"../data/MUTAG/{foldername}/MUTAG_A.txt", delimiter=','))
                .t().long())
        node_to_graph_ind = (torch.from_numpy(np.loadtxt(f"../data/MUTAG/{foldername}/MUTAG_graph_indicator.txt"))
                .squeeze().long())
        node_labels_in_once = (torch.from_numpy(np.loadtxt(f"../data/MUTAG/{foldername}/MUTAG_node_labels.txt"))
                .squeeze().long())
        y = (torch.from_numpy(np.loadtxt(f"../data/MUTAG/{foldername}/MUTAG_graph_labels.txt"))
                .squeeze().long())

        return edge_index, node_to_graph_ind, node_labels_in_once, y
