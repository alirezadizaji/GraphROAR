from dig.xgraph.models import GNNBasic
from torch import nn
from torch_geometric.data import Batch


class GNNWrapper(nn.Module):
    """ the only point of this class is to pass `x` and `edge_index` as separate input
    arguments and therefore torch hooks could recognize them"""
    
    def __init__(self, model: GNNBasic) -> None:
        super().__init__()

        self._model: GNNBasic = model

    def forward(self, data: Batch):
        return self._model(data=data[0])