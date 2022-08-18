from contextlib import contextmanager
import os
from typing import TYPE_CHECKING

import torch

from ...models.gnn_wrapper import GNNWrapper
from .train import TrainEntrypoint
from ...utils.edge_elimination import Arguments, edge_elimination_hook

if TYPE_CHECKING:
    from ...config import ROARConfig
    from dig.xgraph.models import GNNBasic
    from torch_geometric.data import Batch

class ROAREntrypoint(TrainEntrypoint):
    def __init__(self, conf: 'ROARConfig', model: 'GNNBasic') -> None:
        model = GNNWrapper(model)
        self.roar_ratio: float = None
        super().__init__(conf, model)

    def _model_forwarding(self, data: 'Batch') -> torch.Tensor:
        x = self.model([data])
        return x
    
    def _get_weight_save_path(self, epoch_num):
        weight_save_path = super()._get_weight_save_path(epoch_num)
        dirname = os.path.dirname(weight_save_path)
        filename = os.path.basename(weight_save_path)
        new_dirname = os.path.join(dirname, str(self.roar_ratio))
        os.makedirs(new_dirname, exist_ok=True)

        weight_save_path = os.path.join(new_dirname, filename)
        return weight_save_path

    @contextmanager
    def eliminate_edges(self):
        conf: 'ROARConfig' = self.conf
        handle = self.model.register_forward_pre_hook(
                edge_elimination_hook(Arguments(conf.edge_masks_load_dir, self.roar_ratio, conf.edge_mask_symmetric)))
        
        yield None

        handle.remove()

    def run(self):
        conf: 'ROARConfig' = self.conf

        for self.roar_ratio in conf.roar_ratios:
            print(f"*** ROAR {self.roar_ratio*100}% ***", flush=True)
            with self.eliminate_edges():
                super().run()

