import matplotlib.pyplot as plt
import networkx as nx

def visualization(graph, title: str):
    pos = nx.kamada_kawai_layout(graph)

    nx.draw_networkx_nodes(graph, pos,
                        node_size=300)

    nx.draw_networkx_edges(graph, pos, width=3, arrows=False)

    nx.draw_networkx_edges(graph, pos,
        width=6,
        arrows=False)
    
    plt.title(title)
    plt.show()