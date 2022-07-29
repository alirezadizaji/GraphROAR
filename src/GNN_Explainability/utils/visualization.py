import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def visualization(data: Data, title: str):
    graph = to_networkx(data)
    pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph, pos,
                        node_size=300)

    nx.draw_networkx_edges(graph, pos, width=3, arrows=False)

    nx.draw_networkx_edges(graph, pos,
        width=6,
        arrows=False)
    nx.draw_networkx_labels(graph, pos)
    
    plt.title(title)
    plt.show()