from contextlib import contextmanager
import os
from typing import TYPE_CHECKING

import torch

from GNN_Explainability.models.gnn_wrapper import GNNWrapper

from .train import TrainEntrypoint
from ...utils.edge_elimination import edge_elimination

if TYPE_CHECKING:
    from ...config import ROARConfig
    from dig.xgraph.models import GNNBasic
    from torch_geometric.data import Batch

class ROAREntrypoint(TrainEntrypoint):
    def __init__(self, conf: 'ROARConfig', model: 'GNNBasic') -> None:
        model = GNNWrapper(model)
        super().__init__(conf, model)

    def _model_forwarding(self, data: 'Batch') -> torch.Tensor:
        x = self.model([data])
        return x
    
    def _save_model_weight(self, name: str) -> None:
        wrapper: GNNWrapper = self.model
        torch.save(wrapper.model.state_dict(), os.path.join(self.conf.save_dir, f'{name}.pt'))

    @contextmanager
    def eliminate_edges(self, roar_ratio: float):
        conf: 'ROARConfig' = self.conf
        handle = self.model.register_forward_pre_hook(
                edge_elimination(conf.edge_masks_load_dir, roar_ratio))
        
        yield None

        handle.remove()

    def run(self):
        conf: 'ROARConfig' = self.conf

        for ratio in conf.roar_ratios:
            print(f"*** ROAR {ratio*100}% ***", flush=True)
            with self.eliminate_edges(ratio):
                super().run()
            self._save_model_weight(f'model_roar{ratio*100}%')

