import argparse
import random
from sys import argv
import os.path as osp
import os

from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.evaluation import XCollector, ExplanationProcessor
from dig.xgraph.method import GNNExplainer
from dig.xgraph.models import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import DataLoader, InMemoryDataset
from torch_geometric.data import download_url, extract_zip

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(12345)


def index_to_mask(index: torch.Tensor, size: int) -> torch.Tensor:
        mask = torch.zeros(size, dtype=torch.bool, device=index.device)
        mask[index] = 1
        return mask


def split_dataset(dataset: InMemoryDataset,
        num_classes: int,
        train_percent: float=0.7) -> InMemoryDataset:

    indices = []

    for i in range(num_classes):
        index = (dataset.data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)

    rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    dataset.data.train_mask = index_to_mask(train_index, size=dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=dataset.data.num_nodes)

    dataset.data, dataset.slices = dataset.collate([dataset.data])

    return dataset


def check_checkpoints(root='../'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)


if __name__ == "__main__":
    device = torch.device(f'cuda:{argv[1]}' if torch.cuda.is_available() else 'cpu')

    dataset = SynGraphDataset('../datasets', 'BA_shapes')
    dataset.data.x = dataset.data.x.to(dtype=torch.float32, device=device)
    dataset.data.edge_index = dataset.data.edge_index.to(device=device)
    dataset.data.x = dataset.data.x[:, :1]
    data_y = dataset.data.y.clone()
    data_y[dataset.data.y==1] = 2
    data_y[dataset.data.y==2] = 3
    data_y[dataset.data.y==3] = 1
    dataset.data.y = data_y.to(device=device)
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    splitted_dataset = split_dataset(dataset, dataset.data.y.unique().numel())
    splitted_dataset.data.mask = splitted_dataset.data.test_mask
    splitted_dataset.slices['mask'] = splitted_dataset.slices['test_mask']
    dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)

    model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
    model.to(device)
    check_checkpoints()
    ckpt_path = osp.join('../checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])

    raw_pred = model(dataset.data.x, dataset.data.edge_index)
    explainer = GNNExplainer(model, epochs=100, lr=0.01, explain_graph=False)

    sparsities = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    fidelities = []

    for sparsity in sparsities:
        x_collector = XCollector(sparsity)

        for i, data in enumerate(dataloader):
            for j, node_idx in enumerate(torch.where(data.test_mask)[0]):
                y_gt = dataset.data.y[node_idx]
                
                print(f'{j}: explain graph {i} node {node_idx} gt label {y_gt}', flush=True)
                data.to(device)

                if torch.isnan(data.y[0].squeeze()):
                    continue
                
                node_idx = torch.tensor([node_idx])
                edge_masks, hard_edge_masks, related_preds = \
                    explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes, node_idx=node_idx)

                plt.title(f'GNNExplainer (sparsity {sparsity}): node {node_idx.item()}, label {y_gt}')
                explainer.visualize_graph(node_idx, data.edge_index, hard_edge_masks[y_gt], y=data.y, threshold=0.01, nolabel=False)
                save_dir = f"../results/sparsity{sparsity}"
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(osp.join(save_dir, f"{node_idx.item()}.png"))

                x_collector.collect_data(hard_edge_masks, related_preds, y_gt.cpu())

        fidelities.append(x_collector.fidelity)        
        print(f'***sparsity {sparsity}***\nFidelity: {x_collector.fidelity:.4f}\n'
            f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
            f'Sparsity: {x_collector.sparsity:.4f}')
    
    plt.title('BA-Shapes test set')
    plt.xlabel('Sparsity')
    plt.ylabel('Fidelity')
    plt.plot(sparsities, fidelities)
    plt.savefig('../results/sp_fi.png')