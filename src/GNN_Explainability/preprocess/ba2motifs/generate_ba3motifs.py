from copy import deepcopy
from sys import argv
from typing import List, TYPE_CHECKING

from torch_geometric.data import DataLoader

from ...enums import *
from ...dataset.ba_2motifs import BA2MotifsDataset
from ..parse_graphs_to_txt import parse_graphs_to_text
from .triangle_motif_generation import create_new_graphs_with_triangles_motif

if TYPE_CHECKING:
    from torch_geometric.data import Data


if __name__ == "__main__":
    train = DataLoader(BA2MotifsDataset(DataSpec.TRAIN), 
                    batch_size=1)
    val = DataLoader(BA2MotifsDataset(DataSpec.VAL), 
                    batch_size=1)
    test = DataLoader(BA2MotifsDataset(DataSpec.TEST),
                    batch_size=1)
    
    # take all graphs existed in BA2Motifs
    graphs: List['Data'] = list()
    for loader in [train, val, test]:
        for data in loader:
            d = data[0].to_data_list()[0]
            graphs.append(d)
    
    graphs_label1 = list(filter(lambda x: x.y.item() == 1, graphs))
    graphs_label2 = create_new_graphs_with_triangles_motif(deepcopy(graphs_label1))

    graphs = graphs + graphs_label2
    
    # save new generated graphs in text format
    parse_graphs_to_text(graphs, save_dir=argv[1], dataset_name=argv[2])