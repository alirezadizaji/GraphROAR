from abc import abstractmethod, ABC
import os
from typing import TYPE_CHECKING

import numpy as np

from . import ExplainerEntrypoint

if TYPE_CHECKING:
    from torch_geometric.data import Data
    from .....config import ExplainConfig
    from dig.xgraph.models import GNNBasic

class InstanceX(ExplainerEntrypoint, ABC):
    """ Entrypoints for instance-based explainers """

    def __init__(self, conf: 'ExplainConfig', model: 'GNNBasic', explainer) -> None:
        if self.conf.training_config.batch_size > 1:
            print("** WARNING: for instance-based explainer, batch size is changed to one", flush=True)
            self.conf.training_config.batch_size = 1

        super().__init__(conf, model, explainer)

    @abstractmethod
    def explain_instance(self, data: 'Data'):
        pass

    def run(self):
        conf: 'ExplainConfig' = self.conf

        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            for data in loader:
                data: 'Data' = data[0].to(self.conf.device)
                out_x = self.explain_instance(data)
                edge_mask = self.get_edge_mask(out_x)

                if conf.edge_mask_save_dir is not None:
                    if not hasattr(data, 'name'):
                        raise Exception('data object must have name attribute.')
                    save_dir = os.path.join(conf.edge_mask_save_dir, f"{data.name}.npy")
                    np.save(save_dir, edge_mask.cpu().numpy())
                
                self.last_step_todo(out_x)
                self.visualize_sample()
                