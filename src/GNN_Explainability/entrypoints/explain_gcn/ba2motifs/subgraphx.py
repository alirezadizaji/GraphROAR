import argparse
import random
import os.path as osp
import os

from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method import SubgraphX
from dig.xgraph.models import *
from dig.xgraph.method.subgraphx import PlotUtils
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


def check_checkpoints(root='./'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # instantiate dataset
    dataset = SynGraphDataset('./datasets', 'BA_shapes')
    dataset.data.x = dataset.data.x.to(torch.float32)
    dataset.data.x = dataset.data.x[:, :1]
    data_y = dataset.data.y.clone()
    data_y[dataset.data.y==1] = 2
    data_y[dataset.data.y==2] = 3
    data_y[dataset.data.y==3] = 1
    dataset.data.y = data_y
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    splitted_dataset = split_dataset(dataset, dataset.data.y.unique().numel())
    dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)

    # instantiate base model
    model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
    model.to(device)
    check_checkpoints()
    ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])

    # train explainer
    explainer = SubgraphX(model, num_classes=4, device=device, explain_graph=False,
                        reward_method='nc_mc_l_shapley')
    # Visualization
    max_nodes = 15
    node_indices = torch.where(dataset.data.test_mask * dataset.data.y != 0)[0].tolist()
    print(node_indices)
    node_idx = 515
    print(f'explain graph node {node_idx}')
    dataset.data.to(device)
    logits = model(dataset.data.x, dataset.data.edge_index)
    prediction = logits[node_idx].argmax(-1).item()
    print(prediction)
    _, explanation_results, related_preds = \
        explainer(dataset.data.x, dataset.data.edge_index, node_idx=node_idx, max_nodes=max_nodes)

    plotutils = PlotUtils(dataset_name='ba_shapes')
    print(related_preds)
    explainer.visualization(explanation_results[0],
                            max_nodes=max_nodes,
                            plot_utils=plotutils,
                            y=dataset.data.y)