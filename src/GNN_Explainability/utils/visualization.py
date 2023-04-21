import os
from typing import Dict, List, Optional, Tuple, Protocol

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch import Tensor, nonzero
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from ..enums.color import Color

class NodeColor(Protocol):
    def __call__(self, node: 'Tensor') -> Color:
        ...

def visualization(data: Data, 
        title: str,
        pos: Optional[Dict[int, List[int]]] = None,
        save_dir: Optional[str] = None,
        node_color_setter: Optional[NodeColor] = None,
        legend: Optional[Dict[Color, str]] = None,
        node_size: int = 300,
        edge_width: int = 6,
        draw_node_labels: bool = True,
        plot: bool = True,
        edges_color: Optional[List[Color]] = None):
    
    graph = to_networkx(data)

    # Networkx re-orders the edges :(, Therefore edge coloring must be re-ordered too.
    e1 = Tensor(np.array(graph.edges())).unsqueeze(1)
    e2 = data.edge_index.T.unsqueeze(0)
    m = nonzero((e1[..., 0] == e2[..., 0]) & (e1[..., 1] == e2[..., 1]), as_tuple=True)[1]
    if isinstance(edges_color, list):
        edges_color = [edges_color[i] for i in m]

    # set color of nodes
    if node_color_setter is not None:
        node_colors: List[Color] = list()
        for x in data.x:
            color = node_color_setter(x) or Color.DEFAULT
            node_colors.append(color)
    else:
        node_colors = Color.DEFAULT

    if pos is None:
        pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph, pos,
        node_size=node_size, node_color=node_colors)

    nx.draw_networkx_edges(graph, pos,
        width=edge_width,
        arrows=False,
        edge_color=edges_color or 'k')

    if draw_node_labels:
        nx.draw_networkx_labels(graph, pos)
    
    plt.title(title)

    # set legend
    if legend is not None:
        if not isinstance(node_colors, list):
            node_colors = [node_colors]
       
        for color, label in legend.items():
            plt.scatter([],[], c=color, label=label)
        plt.legend()
    
    if plot:
        # save plot if neccessary
        if save_dir is not None:
            save_dir = os.path.join(save_dir, title)
            plt.savefig(save_dir, dpi=600)
        else:
            plt.show()

    return pos