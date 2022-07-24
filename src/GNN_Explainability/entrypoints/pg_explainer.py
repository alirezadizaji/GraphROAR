import argparse
import random
import os.path as osp
import os

from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.evaluation import XCollector, ExplanationProcessor
from dig.xgraph.method import PGExplainer
from dig.xgraph.models import *
from dig.xgraph.method.pgexplainer import PlotUtils
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
    splitted_dataset.data.mask = splitted_dataset.data.test_mask
    splitted_dataset.slices['mask'] = splitted_dataset.slices['test_mask']
    dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)

    # instantiate base model
    model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
    model.to(device)
    check_checkpoints()
    ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])

    # train explainer
    explainer = PGExplainer(model, in_channels=900, device=device, explain_graph = False, epochs = 5, k=3)
    # explainer.train_explanation_network(splitted_dataset)
    # torch.save(explainer.state_dict(), 'tmp.pt')
    state_dict = torch.load('tmp.pt')
    explainer.load_state_dict(state_dict)

    node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()

    # # visualization
    # plotutils = PlotUtils(dataset_name='ba_shapes', is_show=True)
    # data = dataset[0]
    # node_idx = node_indices[6]
    # with torch.no_grad():
    #     walks, masks, related_preds = \
    #         explainer(data.x, data.edge_index, node_idx=node_idx, y=data.y, top_k=5)

    # explainer.visualization(data, edge_mask=masks[0], top_k=5, plot_utils=plotutils, node_idx=node_idx)
    
    # get metrics
    sparsities = []
    fidelities = []
    for top_k in [1, 2, 3, 4, 5]:
        undirected_graph = True
        x_collector = XCollector()
        index = -1
        node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
        top_k = top_k if not undirected_graph else top_k * 2

        for i, data in enumerate(dataloader):
            for j, node_idx in enumerate(node_indices):
                index += 1
                print(index)
                data.to(device)

                if torch.isnan(data.y[0].squeeze()):
                    continue

                with torch.no_grad():
                    walks, masks, related_preds = \
                        explainer(data.x, data.edge_index, node_idx=node_idx, y=data.y, top_k=top_k)
                    masks = [mask.detach() for mask in masks]
                x_collector.collect_data(masks, related_preds)

                # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
                # obtain the result: x_processor(data, masks, x_collector)

        sparsities.append(x_collector.sparsity)
        fidelities.append(x_collector.fidelity)
        print(f'Fidelity: {x_collector.fidelity:.4f}\n'
            f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
            f'Sparsity: {x_collector.sparsity:.4f}')
    
    plt.title('PGExplainer BA shapes test set')
    plt.xlabel('sparsity')
    plt.ylabel('fidelity')
    plt.plot(sparsities, fidelities)
    plt.show()