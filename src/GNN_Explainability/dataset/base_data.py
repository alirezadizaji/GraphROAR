from typing import List

from torch_geometric.data import Data

class BaseData(Data):
    def __init__(self, name: List[str], x=None, edge_index=None, edge_attr=None,
            y=None, pos=None, normal=None, face=None, **kwargs):
        
        super().__init__(x, edge_index, edge_attr, y, pos, normal, face, **kwargs) 
        
        self.name = name