import os
from typing import List, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data

def parse_graphs_to_text(graphs: List['Data'], 
        save_dir: str,
        dataset_name: str) -> None:
    """ It gets a list of graphs and parses them to three separate text files:
    - *_A.txt: how nodes are connected with each other
    - *_graph_indicator.txt: each node belongs to which graph id
    - *_graph_labels.txt: label of each graph
    """
    os.makedirs(save_dir, exist_ok=True)

    file_A_name = f"{dataset_name}_A.txt"
    file_graph_labels = f"{dataset_name}_graph_labels.txt"
    file_indicator_name = f"{dataset_name}_graph_indicator.txt"

    node_starting_id = 1
    labels: List[int] = list()

    for i, g in enumerate(graphs):
        labels.append(g.y.item())
        
        # order node indices to start from zero
        indices = g.edge_index - g.edge_index.min()

        indices = indices + node_starting_id
        num_nodes = torch.unique(indices).numel()
        node_starting_id = node_starting_id + num_nodes
        
        with open(os.path.join(save_dir, file_A_name), 'ab') as f:
            np.savetxt(f, indices.cpu().numpy().T, delimiter=', ', fmt='%d')
        
        with open(os.path.join(save_dir, file_graph_labels), 'ab') as f:
            np.savetxt(f, g.y.unsqueeze(0).cpu().numpy(), fmt='%d')

        with open(os.path.join(save_dir, file_indicator_name), 'ab') as f:
            graph_inds = [i + 1] * num_nodes
            np.savetxt(f, np.array(graph_inds), fmt='%d')