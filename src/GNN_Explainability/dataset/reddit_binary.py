from random import shuffle
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from .base_dataset import GraphClsDataset
from ..enums.data_spec import DataSpec

if TYPE_CHECKING:
    from torch import Tensor

class RedditDataset(GraphClsDataset):
    def _read_file(self) -> Tuple['Tensor', 'Tensor', 'Tensor', 'Tensor']:
        edge_index = (torch.from_numpy(np.loadtxt(f"../data/REDDIT-BINARY/raw/REDDIT-BINARY_A.txt", delimiter=','))
                .t().long())
        node_to_graph_ind = (torch.from_numpy(np.loadtxt(f"../data/REDDIT-BINARY/raw/REDDIT-BINARY_graph_indicator.txt"))
                .squeeze().long())
        y = (torch.from_numpy(np.loadtxt(f"../data/REDDIT-BINARY/raw/REDDIT-BINARY_graph_labels.txt"))
                .squeeze().long())

        # All nodes are unique and therefore have identical features
        node_labels_in_once = torch.zeros_like(node_to_graph_ind)

        return edge_index, node_to_graph_ind, node_labels_in_once, y
