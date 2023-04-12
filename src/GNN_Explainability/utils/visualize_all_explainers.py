""" This script aims to visualize explainer outputs on randomly selected
graphs. """

from argparse import ArgumentParser, Namespace
from copy import deepcopy
import os
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import DataLoader

from ..dataset.mutag import MUTAGDataset
from ..dataset.reddit_binary import RedditDataset
from ..entrypoints.core.main import get_dataset_cls
from ..enums.data_spec import DataSpec
from .symmetric_edge_mask import symmetric_edges
from .visualization import visualization

def get_parser() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-D', '--dataset', type=str, help='dataset name to visualize')
    parser.add_argument('-E', '--edge-mask-dir', type=str, help='edge mask directory to get probability weightings from')
    parser.add_argument('-R', '--ratio', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9], help='ratios to visualize (must be between 0.0 and 1.0)')
    parser.add_argument('-X', '--explainer', type=str, nargs='+', default=['random', 'gnnexplainer', 'gradcam', 'pgexplainer', 'subgraphx'], help='explainers to visualize')
    parser.add_argument('-N', '--num-samples', type=int, default=20, help='number of (randomly selected) samples to visualize.')
    parser.add_argument('-S', '--symetric', action='store_true', help='If passed then edge weights must be symmetric.')
    parser.add_argument('-W', '--width', type=int, default=3, help='width of a subplot')
    parser.add_argument('-H', '--height', type=int, default=2.5, help='height of a subplot')
    parser.add_argument('-V', '--save-dir', type=str, help='root directory to save visualizations.')
    parser.add_argument('-I', '--node-size', type=int, default=15, help='node size to be depicted.')
    parser.add_argument('-M', '--max-nodes', type=int, default=60, help='maximum number of nodes a graph could have to be considered as a candidate for visualization.')
    parser.add_argument('-O', '--show-legend-once', action='store_true', help='If given then the legend will appear only on graph `org` subplot.')
    parser.add_argument('-C', '--edge-color', type=str, default='#2E7D32', help='Edge color of explanation outputs.')
    parser.add_argument('-L', '--list-names', type=str, nargs='+', help='The list to be depicted on them only.')
    args = parser.parse_args()
    return args

def get_color_setter_and_legend(cls):
    if cls == MUTAGDataset:
        from ..entrypoints.explain_gcn.mutag.gin3l.gnn_explainer import color_setter, legend
    else:
        color_setter = None
        legend = None
    
    return color_setter, legend

def visualize(args: Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    cls = get_dataset_cls(args.dataset)
    color_setter, legend = get_color_setter_and_legend(cls)

    train_set = cls(DataSpec.TRAIN)
    val_set = cls(DataSpec.VAL)
    test_set = cls(DataSpec.TEST)

    train = DataLoader(train_set)
    val = DataLoader(val_set)
    test = DataLoader(test_set)

    graphs = {}
    for loader in [train, val, test]:
        for data in loader:
            data = data[0]
            # graphs with more than a threshold number of nodes could not be clearly visualized as the plot becomes untidy.
            if data.x.size(0) > args.max_nodes:
                continue
            graphs[data.name[0]] = data
    
    if args.list_names is not None:
        samples_names = args.list_names
    else:
        samples_names = sample(graphs.keys(), args.num_samples)
    
    for name in samples_names:
        n_row, n_col = len(args.explainer), len(args.ratio)
        w_size, h_size = n_col * args.width, n_row * args.height
        fig = plt.figure(figsize=(w_size, h_size))
        subplot_legend = legend

        graph = graphs[name]
        
        # just to keep all positions of the graphs the same
        pos = None
        
        for row_ix, explainer in enumerate(args.explainer):
            
            if explainer != 'subgraphx':
                edge_weights_dir = os.path.join(args.edge_mask_dir, explainer, f"{name}.npy")
                weights = torch.from_numpy(np.load(edge_weights_dir))
                weights = symmetric_edges(graph.edge_index, weights)
                num_edges = weights.numel()

            for col_ix, r in enumerate(args.ratio):

                if explainer == 'subgraphx':
                    if r == 1.0:
                        weights = torch.ones(graph.edge_index.size(1))
                    else:
                        edge_weights_dir = os.path.join(args.edge_mask_dir, f"{explainer}_{int(r*100)}%", f"{name}.npy")
                        weights = torch.from_numpy(np.load(edge_weights_dir))
                        weights = symmetric_edges(graph.edge_index, weights)
                    num_edges = weights.numel()

                ix = row_ix * n_col + col_ix + 1

                # first column is only for original graph
                # if col_ix == 0 and row_ix > 0:
                #     continue

                fig.add_subplot(n_row, n_col, ix)

                k = int(r * num_edges)
                k = k - k % 2
                _, inds = torch.topk(weights, k)
                mask = torch.zeros_like(weights).bool()
                mask[inds] = True

                g = deepcopy(graph)

                if r == 1.0:
                    edge_colors = None
                else:
                    edge_colors = [args.edge_color if mask[i] else '#C6C6C6' for i in range(mask.numel())]
                # g.edge_index = g.edge_index[:, mask]

                new_pos = visualization(g, 
                    title='', 
                    pos=pos, 
                    node_size=args.node_size, 
                    edge_width=3, 
                    node_color_setter=color_setter,
                    legend=subplot_legend,
                    draw_node_labels=False, 
                    plot=False,
                    edges_color=edge_colors)

                if args.show_legend_once and col_ix == 0 and row_ix == 0:
                    subplot_legend = None
                    
                if row_ix == 0:
                    title = "org" if r == 1.0 else f"{int(r*100)}%"
                    plt.title(title, fontsize=20)

                if col_ix == 0:
                    plt.ylabel(explainer, fontsize=20)

                if pos is None:
                    pos = new_pos

        save_dir = os.path.join(args.save_dir, f"{name}_{graph.y.item()}.png")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save_dir, dpi=600, bbox_inches='tight')
        print(f'DONE: {name} saved at {save_dir}.', flush=True)

if __name__ == "__main__":
    args = get_parser()
    visualize(args)
