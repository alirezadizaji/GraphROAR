from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

import torch 

from ...main import MainEntrypoint
from .....utils.decorators.func_counter import counter
from .....utils.visualization import visualization

if TYPE_CHECKING:
    from .....config import ExplainConfig
    from torch_geometric.data import Data
    from dig.xgraph.models import GNNBasic

class ExplainerEntrypoint(MainEntrypoint, ABC):
    def __init__(self, conf: 'ExplainConfig', model: 'GNNBasic', explainer) -> None:
        super().__init__(conf, model)

        self._explainer = explainer
        """ explainer method """

    def get_model_pred(self, data: 'Data'):
        y_pred = self.model(data=data).argmax(-1)
        return y_pred

    @abstractmethod
    def last_step_todo(self, out_x):
        """ Anything needs to do at last with explainer output """
        pass
    
    @abstractmethod
    def get_edge_mask(self, out_x) -> torch.Tensor:
        pass

    @counter(0)
    def visualize_sample(self, data: 'Data', edge_mask: torch.Tensor):
        conf: 'ExplainConfig' = self.conf
        if self.visualize_sample.call > conf.num_instances_to_visualize:
            return

        if not hasattr(data, 'name'):
            raise Exception('The data instance must have `name` attribute.')

        name = data.name
        save_dir = conf.save_dir if conf.save_visualization else None
        pos = visualization(data, f'{name}_org', save_dir=save_dir)
        data.edge_index = data.edge_index[:, edge_mask >= 0.5]
        visualization(data, f"{name}_X", pos, save_dir)