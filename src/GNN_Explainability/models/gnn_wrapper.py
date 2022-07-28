from dig.xgraph.models import GNNBasic
from torch import nn

class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

class GNNWrapper(nn.Module):
    """ the only point of this class is to pass `x` and `edge_index` as separate input
    arguments and therefore torch hooks could recognize them"""
    
    def __init__(self, model: GNNBasic) -> None:
        super().__init__()

        self.identity: Identity = Identity()
        self._model: GNNBasic = model

    def forward(self, x, edge_index, *args, **kwargs):
        edge_index = self.identity(edge_index)
        kwargs['x'] = x
        kwargs['edge_index'] = edge_index
        return self._model(*args, **kwargs)