from typing import List

from dig.xgraph.models import GNNBasic
from torch import nn
from torch_geometric.data import Batch


class GNNWrapper(nn.Module):
    r""" The point of this class is to only enable using hooks with DIG models as 
    most of which do not support hooking due to using ONLY keyword arguments in `forward` function."""
    
    def __init__(self, model: 'GNNBasic') -> None:
        super().__init__()

        self.model: 'GNNBasic' = model

    def forward(self, data: List[Batch]):
        return self.model(data=data[0])