from contextlib import contextmanager
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...models.gnn_wrapper import GNNWrapper
from .train import TrainEntrypoint
from ...utils.edge_elimination import EdgeEliminatorArgs, edge_elimination_hook
from ...utils.seed_changer import seed_changer

if TYPE_CHECKING:
    from ...config import RetrainingConfig
    from dig.xgraph.models import GNNBasic
    from torch_geometric.data import Batch

class RetrainingEntrypoint(TrainEntrypoint):
    def __init__(self, conf: 'RetrainingConfig', model: 'GNNBasic') -> None:
        model = GNNWrapper(model)
        self.retraining_ratio: float = None
        super().__init__(conf, model)

    def _model_forwarding(self, data: 'Batch') -> torch.Tensor:
        x = self.model([data])
        return x
    
    def _get_weight_save_path(self, epoch_num):
        weight_save_path = super()._get_weight_save_path(epoch_num)
        dirname = os.path.dirname(weight_save_path)
        filename = os.path.basename(weight_save_path)
        new_dirname = os.path.join(dirname, str(self.retraining_ratio))
        os.makedirs(new_dirname, exist_ok=True)

        weight_save_path = os.path.join(new_dirname, filename)
        return weight_save_path


    def _create_random_edge_mask(self):
        print("@@@ start random edge mask creation @@@", flush=True)
        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            for batch_data in loader:
                batch_data = batch_data[0]
                
                B = batch_data.y.numel()
                for i in range(B):
                    data = batch_data[i]
                    num_edges = data.edge_index.size(1)
                    rand_edge_mask = torch.rand(num_edges)
                    os.makedirs(self.conf.edge_masks_load_dir, exist_ok=True)
                    path = os.path.join(self.conf.edge_masks_load_dir, f"{data.name}.npy")
                    np.save(path, rand_edge_mask)

        print("random edge mask creation done.", flush=True)


    @contextmanager
    def eliminate_edges(self):
        conf: 'RetrainingConfig' = self.conf
        handle = self.model.register_forward_pre_hook(
                edge_elimination_hook(
                    EdgeEliminatorArgs(conf.edge_masks_load_dir,
                        self.retraining_ratio,
                        conf.edge_mask_symmetric,
                        conf.eliminate_top_most_edges,
                        conf.eliminate_nodes_too),
                    skip_during_eval=conf.skip_during_evaluation))
        
        yield None

        handle.remove()

    def run(self):
        conf: 'RetrainingConfig' = self.conf

        if conf.edge_mask_random_weighting:
            with seed_changer():
                self._create_random_edge_mask()

        for self.retraining_ratio in conf.retraining_ratios:
            print(f"*** Retraining {self.retraining_ratio*100}% ***", flush=True)
            with self.eliminate_edges():
                super().run()

