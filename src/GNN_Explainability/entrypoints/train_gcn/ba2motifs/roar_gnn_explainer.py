from tqdm import tqdm

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, Data

from ....dataset.ba_2motifs import BA2MotifsDataset
from ....enums.data_spec import DataSpec
from ...main import MainEntrypoint
from ....utils.edge_elimination import edge_elimination

class Entrypoint(MainEntrypoint):
    def run():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_epochs = 10
        train_loader = DataLoader(BA2MotifsDataset(DataSpec.TRAIN), shuffle=False)
        val_loader = DataLoader(BA2MotifsDataset(DataSpec.VAL))

        model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        model.to(device)
        
        optimizer = Adam(model.parameters(), lr=1e-4)
        total_loss = []

        for percentage in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for epoch in tqdm(range(num_epochs)):
                model.train()

                correct = 0
                total = 0
                handle = model.register_forward_pre_hook(
                    edge_elimination('./GNN_Explainability/data/ba2motifs', 'train', percentage))
                for i, data in enumerate(train_loader):
                    data: Data = data[0].to(device)
                    x = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
                    x = F.log_softmax(x, dim=-1)
                    loss = - torch.mean(x[torch.arange(x.size(0)), data.y])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss.append(loss.item())
                    if i % 20 == 0:
                        print(f'epoch {epoch} iter {i} loss value {np.mean(total_loss)}')
                    y_pred = x.argmax(-1)
                    correct += torch.sum(y_pred == data.y).item()
                    total += data.y.numel()
                
                print(f'roar {percentage}% epoch {epoch} train acc {correct/total}')
                handle.remove()


                handle = model.register_forward_pre_hook(
                    edge_elimination('./GNN_Explainability/data/ba2motifs', 'val', percentage))
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in val_loader:
                        data: Data = data[0].to(device)
                        x = model(x=data.x, edge_index=data.edge_index)
                        x = F.log_softmax(x, dim=-1)
                        y_pred = x.argmax(-1)
                        correct += torch.sum(y_pred == data.y).item()
                        total += data.y.numel()
                    print(f'roar {percentage}% epoch {epoch} val acc {correct/total}')

            torch.save(model.state_dict(), './GNN_Explainability/checkpoints/ba2motifs_gin_3l.pt')
