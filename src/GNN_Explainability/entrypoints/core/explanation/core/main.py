from abc import abstractmethod, ABC
import os
from typing import TYPE_CHECKING, Generic, TypeVar

import matplotlib.pyplot as plt
import torch 

from ...main import MainEntrypoint
from .....utils.decorators.func_counter import counter
from .....utils.visualization import visualization
from .....utils.symmetric_edge_mask import symmetric_edges 

if TYPE_CHECKING:
    from .....config import ExplainConfig
    from torch_geometric.data import Data
    from dig.xgraph.models import GNNBasic

class ExplainerEntrypoint(MainEntrypoint, ABC):
    @abstractmethod
    def get_edge_mask(self, out_x, data: 'Data') -> torch.Tensor:
        pass

    
    def _select_explainable_edges(self, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        edge_mask = symmetric_edges(edge_index, edge_mask)
        edge_index = edge_index[:, edge_mask >= 0.5]
        return edge_index


    @counter(0)
    def visualize_sample(self, data: 'Data', edge_mask: torch.Tensor):
        conf = self.conf
        if self.visualize_sample.call > conf.num_instances_to_visualize:
            return

        if not hasattr(data, 'name'):
            raise Exception('The data instance must have `name` attribute.')

        name = data.name
        if conf.save_visualization:
            save_dir = os.path.join(conf.save_dir, 'visualizations')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = None
        pos = visualization(data,
            f'{name[0]}_org_{data.y.item()}',
            save_dir=save_dir,
            node_color_setter=self.conf.node_color_setter,
            legend=self.conf.plt_legend)
        plt.close()
        
        data.edge_index = self._select_explainable_edges(data.edge_index, edge_mask)
        visualization(data,
            f"{name[0]}_X_{data.y.item()}",
            pos,
            save_dir,
            node_color_setter=self.conf.node_color_setter,
            legend=self.conf.plt_legend)
        plt.close()