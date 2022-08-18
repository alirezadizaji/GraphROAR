import math
import os
from tqdm import tqdm
from typing import TYPE_CHECKING

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F

from ...enums.data_spec import DataSpec

from . import MainEntrypoint

if TYPE_CHECKING:
    from torch_geometric.data import Batch, DataLoader
    from ...config import BaseConfig


class TrainEntrypoint(MainEntrypoint):

    def _get_weight_save_path(self, epoch_num):
        weight_dir_name = 'weights'
        weight_save_dir =  os.path.join(self.conf.save_dir, weight_dir_name)
        os.makedirs(weight_save_dir, exist_ok=True)
        weight_save_path = os.path.join(weight_save_dir, str(epoch_num))
        return weight_save_path

    def _model_forwarding(self, data: 'Batch') -> torch.Tensor:
        x = self.model(data=data)
        return x

    @staticmethod
    def _calc_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.log_softmax(x, dim=-1)
        loss = - torch.mean(x[torch.arange(x.size(0)), y])
        return loss

    def _save_model_weight(self, epoch_num: int) -> None:
        torch.save(self.model.state_dict(), self._get_weight_save_path(epoch_num))

    def _train_for_one_epoch(self, epoch_num: int) -> None:
        conf: 'BaseConfig' = self.conf

        self.model.train()

        correct = 0
        total = 0        
        total_loss = []

        for i, data in enumerate(self.train_loader):
            data: 'Batch' = data[0].to(conf.device)
            x = self._model_forwarding(data)
            loss = self._calc_loss(x, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss.append(loss.item())
    
            y_pred = x.argmax(-1)
            correct += torch.sum(y_pred == data.y).item()
            total += data.y.numel()
        
            if i % 20 == 0:
                print(f'epoch {epoch_num} iter {i} loss value {np.mean(total_loss)}', flush=True)
    
        print(f'epoch {epoch_num} train acc {correct/total}', flush=True)

    def _validate_for_one_epoch(self, epoch_num: int, loader: 'DataLoader') -> None:
        self.model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            
            for data in loader:
                data: Batch = data[0].to(self.conf.device)
                y_pred = self._model_forwarding(data).argmax(-1)
                correct += torch.sum(y_pred == data.y).item()
                total += data.y.numel()
            
            acc = correct / total
            print(f'epoch{epoch_num} {loader.dataset.dataspec.lower()} acc {acc}', flush=True)
            return acc

    def run(self):
        val_accs = np.zeros(self.conf.training_config.num_epochs)

        for epoch in tqdm(range(self.conf.training_config.num_epochs)):
            self.epoch = epoch
            self._train_for_one_epoch(epoch)  
            self._save_model_weight(epoch)
            val_accs[epoch] = self._validate_for_one_epoch(epoch, self.val_loader)              
        
        # evaluate best epoch on test set
        best_epoch = val_accs.argmax()
        self.model.load_state_dict(torch.load(self._get_weight_save_path(best_epoch), map_location=self.conf.device))
        test_acc =self._validate_for_one_epoch(epoch, self.test_loader)
        val_acc = val_accs[best_epoch]
        print(f"@@@@@ Best accuracy epoch {best_epoch} @@@@@\nval: {val_acc}, test: {test_acc}", flush=True)
