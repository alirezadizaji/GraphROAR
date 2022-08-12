from abc import abstractmethod, ABC
import os
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from . import ExplainerEntrypoint

if TYPE_CHECKING:
    from torch_geometric.data import Data
    from .....config import ExplainConfig
    from dig.xgraph.models import GNNBasic

TExp = TypeVar('TExp')
class InstanceX(ExplainerEntrypoint, Generic[TExp], ABC):
    """ Entrypoints for instance-based explainers """

    def __init__(self, conf: 'ExplainConfig', model: 'GNNBasic', explainer: TExp) -> None:        
        if conf.training_config.batch_size > 1:
            print("** WARNING: for instance-based explainer, batch size is changed to one", flush=True)
            conf.training_config.batch_size = 1
        
        super().__init__(conf, model)        

        self.explainer = explainer
        
    @abstractmethod
    def explain_instance(self, data: 'Data'):
        pass

    def run(self):
        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            for data in loader:
                data: 'Data' = data[0].to(self.conf.device)
                if not hasattr(data, 'name'):
                        raise Exception('data object must have name attribute.')
                print(f"graph {data.name[0]}, label {data.y.item()}", flush=True)

                out_x = self.explain_instance(data)
                edge_mask = self.get_edge_mask(out_x, data)

                if self.conf.edge_mask_save_dir is not None:
                    os.makedirs(self.conf.edge_mask_save_dir, exist_ok=True)
                    save_dir = os.path.join(self.conf.edge_mask_save_dir, f"{data.name[0]}.npy")
                    np.save(save_dir, edge_mask.cpu().numpy())
                
                self.visualize_sample(data, edge_mask)
                