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

class Entrypoint(MainEntrypoint):
    def run():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_epochs = 10
        train_loader = DataLoader(BA2MotifsDataset(DataSpec.TRAIN), shuffle=True)
        val_loader = DataLoader(BA2MotifsDataset(DataSpec.VAL))

        model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        model.to(device)
        
        optimizer = Adam(model.parameters(), lr=1e-4)
        total_loss = []

        for epoch in tqdm(range(num_epochs)):
            model.train()

            correct = 0
            total = 0
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
            
            print(f'epoch {epoch} train acc {correct/total}')
            
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
                print(f'epoch {epoch} val acc {correct/total}')

        torch.save(model.state_dict(), './GNN_Explainability/checkpoints/ba2motifs_gin_3l.pt')
        # for i, data in enumerate(dataloader):
        #     print(data.test_mask.shape)
        #     for j, node_idx in enumerate(torch.where(data.test_mask)[0]):
        #         node_idx = 300
        #         y_gt = dataset.data.y[node_idx]
                
        #         print(f'{j}: explain graph {i} node {node_idx} gt label {y_gt}', flush=True)
        #         data.to(device)

        #         if torch.isnan(data.y[0].squeeze()):
        #             continue
                
        #         node_idx = torch.tensor([node_idx])
        #         edge_masks, hard_edge_masks, related_preds = \
        #             explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes, node_idx=node_idx, target_label=y_gt)

        #         plt.title(f'GNNExplainer: node {node_idx.item()}, label {y_gt}')
        #         explainer.visualize_graph(node_idx, data.edge_index, hard_edge_masks[y_gt], y=data.y, threshold=0.01, nolabel=False)
        #         plt.show()

        #         x_collector.collect_data(hard_edge_masks, related_preds, y_gt)
        #         break

        # print(f'***sparsity {sparsity}***\nFidelity: {x_collector.fidelity:.4f}\n'
        #     f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
        #     f'Sparsity: {x_collector.sparsity:.4f}')
