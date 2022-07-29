from tqdm import tqdm

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, Data


from ....config.base_config import BaseConfig
from ....dataset.ba_2motifs import BA2MotifsDataset
from ....enums.data_spec import DataSpec
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
        conf.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        conf.num_epochs = 10

        conf.base_model = GNNWrapper(
            GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        )
        conf.optimizer = Adam(conf.base_model.parameters(), lr=1e-4)
        self.roar_percentage: float = None

        super(Entrypoint, self).__init__(conf)

    def _train_for_one_epoch(self, epoch_num: int):
        self.conf.base_model.train()

        correct = 0
        total = 0        
        total_loss = []

        handle = self.conf.base_model.identity.register_forward_pre_hook(
            edge_elimination('../data/ba_2motifs/explanation/gnnexplainer', 'train', self.roar_percentage))
        
        for i, data in enumerate(self.conf.train_loader):
            data: Data = data[0].to(self.conf.device)
            x = self.conf.base_model(x=data.x, edge_index=data.edge_index, batch=data.batch)
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
                print(f'epoch {epoch_num} roar {self.roar_percentage*100}% iter {i} loss value {np.mean(total_loss)}', flush=True)
    
        print(f'epoch {epoch_num} roar {self.roar_percentage*100}% train acc {correct/total}', flush=True)
        handle.remove()


    def _validate_for_one_epoch(self, epoch_num: int, loader: DataLoader):
        self.conf.base_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            handle = self.conf.base_model.identity.register_forward_pre_hook(
                edge_elimination('../data/ba_2motifs/explanation/gnnexplainer', loader.dataset.dataspec.lower(), self.roar_percentage))
            
            for data in loader:
                data: Data = data[0].to(self.conf.device)
                x = self.conf.base_model(x=data.x, edge_index=data.edge_index)
                x = F.log_softmax(x, dim=-1)
                y_pred = x.argmax(-1)
            
                correct += torch.sum(y_pred == data.y).item()
                total += data.y.numel()
            
            print(f'epoch{epoch_num} roar {self.roar_percentage}% {loader.dataset.dataspec.lower()} acc {correct/total}', flush=True)
            handle.remove()

    def run(self):
        num_epochs = 10

        for percentage in [0.1, 0.3, 0.5, 0.7, 0.9]:
            self.roar_percentage = percentage

            for epoch in tqdm(range(num_epochs)):
                self.epoch = epoch
                self._train_for_one_epoch(epoch)  
                self._validate_for_one_epoch(epoch, self.conf.val_loader)              

            self._validate_for_one_epoch(epoch, self.conf.test_loader)
            
            torch.save(self.conf.base_model.state_dict(), f'./GNN_Explainability/checkpoints/ba2motifs_roar{percentage}%_gnnexplainer.pt')
