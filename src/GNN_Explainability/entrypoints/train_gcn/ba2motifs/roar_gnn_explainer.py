from tqdm import tqdm

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, Data


from ....config.base_config import BaseConfig
from ....enums.dataset import Dataset
from ...main import MainEntrypoint
from ....models.gnn_wrapper import GNNWrapper
from ....utils.edge_elimination import edge_elimination

class Entrypoint(MainEntrypoint):
    
    def __init__(self):
        conf = BaseConfig(
            try_num=3,
            try_name='roar_gnnexplainer',
            dataset_name=Dataset.BA2Motifs,
        )
        conf.device = torch.device('cpu')
        conf.num_epochs = 100

        conf.save_log_in_file = False
        conf.shuffle_training = True
        self.roar_ratio: float = None

        super(Entrypoint, self).__init__(conf)

    def _init_model_and_optim(self):
        self.conf.base_model = GNNWrapper(
            GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        )
        self.conf.base_model.to(self.conf.device)
        self.conf.optimizer = Adam(self.conf.base_model.parameters(), lr=1e-2)
    
    def _train_for_one_epoch(self, epoch_num: int):
        self.conf.base_model.train()

        correct = 0
        total = 0        
        total_loss = []

        for i, data in enumerate(self.conf.train_loader):
            data: Batch = data[0].to(self.conf.device)
            x = self.conf.base_model([data])
            x = F.log_softmax(x, dim=-1)
            loss = - torch.mean(x[torch.arange(x.size(0)), data.y])
    
            loss.backward()
            self.conf.optimizer.step()
            self.conf.optimizer.zero_grad()

            total_loss.append(loss.item())
    
            y_pred = x.argmax(-1)
            correct += torch.sum(y_pred == data.y).item()
            total += data.y.numel()
        
            if i % 20 == 0:
                print(f'epoch {epoch_num} roar {self.roar_ratio*100}% iter {i} loss value {np.mean(total_loss)}', flush=True)
    
        print(f'epoch {epoch_num} roar {self.roar_ratio*100}% train acc {correct/total}', flush=True)


    def _validate_for_one_epoch(self, epoch_num: int, loader: DataLoader):
        self.conf.base_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            
            for data in loader:
                data: Batch = data[0].to(self.conf.device)
                x = self.conf.base_model([data])
                x = F.log_softmax(x, dim=-1)
                y_pred = x.argmax(-1)
                correct += torch.sum(y_pred == data.y).item()
                total += data.y.numel()
            
            print(f'epoch{epoch_num} roar {self.roar_ratio*100}% {loader.dataset.dataspec.lower()} acc {correct/total}', flush=True)


    def run(self):
        for ratio in [0.1, 0.5, 0.7, 0.9]:
            self._init_model_and_optim()

            self.roar_ratio = ratio
            
            handle = self.conf.base_model.register_forward_pre_hook(
                edge_elimination('../data/ba_2motifs/explanation/gnnexplainer', self.roar_ratio))

            for epoch in tqdm(range(self.conf.num_epochs)):
                self.epoch = epoch
                self._train_for_one_epoch(epoch)  
                self._validate_for_one_epoch(epoch, self.conf.val_loader)              
            self._validate_for_one_epoch(epoch, self.conf.test_loader)
            
            handle.remove()

            torch.save(self.conf.base_model.state_dict(), f'./GNN_Explainability/checkpoints/ba2motifs_roar{ratio*100}%_gnnexplainer.pt')

