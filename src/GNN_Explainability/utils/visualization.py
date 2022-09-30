import os
from typing import Dict, List, Optional, Tuple, Protocol

import matplotlib.pyplot as plt
import networkx as nx
from torch import Tensor
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
        legend: Optional[Dict[Color, str]] = None):
    
    graph = to_networkx(data)
    
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
        node_size=300, node_color=node_colors)

    nx.draw_networkx_edges(graph, pos, width=3, arrows=False)

    nx.draw_networkx_edges(graph, pos,
        width=6,
        arrows=False)
    nx.draw_networkx_labels(graph, pos)
    
    plt.title(title)

    # set legend
    if legend is not None:
        if not isinstance(node_colors, list):
            node_colors_ = [node_colors]

        # exclude legends not exist in the plot
        legend = dict(filter(lambda x: x[0] in node_colors_, legend.items()))
        
        for color, label in legend.items():
            plt.scatter([],[], c=color, label=label)
        plt.legend()
    
    # save plot if neccessary
    if save_dir is not None:
        save_dir = os.path.join(save_dir, title)
        plt.savefig(save_dir, dpi=600)
    else:
        plt.show()

    return pos