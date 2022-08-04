from tqdm import tqdm

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.utils import add_remaining_self_loops

from ....config.base_config import BaseConfig
from ....dataset import BaseData
from ....enums import *
from ...main import MainEntrypoint
from ....utils.visualization import visualization

class Entrypoint(MainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=1,
            try_name='ba2motifs',
            dataset_name=Dataset.BA2Motifs,
        )
        
        conf.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        conf.num_epochs = 100
        
        conf.save_log_in_file = False
        conf.shuffle_training = True
        
        conf.base_model = GCN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        conf.base_model.to(conf.device)
        
        conf.optimizer = Adam(conf.base_model.parameters(), lr=1e-3)
        
        super(Entrypoint, self).__init__(conf)

    def run(self):
        for epoch in tqdm(range(self.conf.num_epochs)):
            self.conf.base_model.train()

            correct = 0
            total = 0
            total_loss = []
            for i, data in enumerate(self.conf.train_loader):
                data=data[0]
                x = self.conf.base_model(data=data)
                x = F.log_softmax(x, dim=-1)
                loss = - torch.mean(x[torch.arange(x.size(0)), data.y])
                loss.backward()
                self.conf.optimizer.step()
                self.conf.optimizer.zero_grad()

                total_loss.append(loss.item())
                if i % 1 == 0:
                    print(f'epoch {epoch} iter {i} loss value {np.mean(total_loss)}')
                y_pred = x.argmax(-1)
                correct += torch.sum(y_pred == data.y).item()
                total += data.y.numel()

            print(f'epoch {epoch} train acc {correct/total}')
            
            with torch.no_grad():
                correct = 0
                total = 0
                for data in self.conf.val_loader:
                    data = data[0]
                    x = self.conf.base_model(data=data)
                    x = F.log_softmax(x, dim=-1)
                    y_pred = x.argmax(-1)
                    correct += torch.sum(y_pred == data.y).item()
                    total += data.y.numel()
                print(f'epoch {epoch} val acc {correct/total}')

        torch.save(self.conf.base_model.state_dict(), './GNN_Explainability/checkpoints/ba2motifs_gin_3l.pt')
