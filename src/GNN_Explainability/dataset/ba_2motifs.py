from typing import Dict, List
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..enums.data_spec import DataSpec


class BA2MotifsDataset(Dataset):
    """a BA2Motifs dataset. """

    def __init__(self, dataspec: DataSpec):
        """
        Args:
            dataspec (DataSpec): data specification for which samples are loaded
        """
        self.dataspec: DataSpec = dataspec

        ba_2motifs = torch.load('../data/ba_2motifs/processed/data.pt')
        self.data: Data = ba_2motifs[0]
        dataset_num = self.data.y.numel()
        
        meta_info: Dict[str, torch.Tensor] = ba_2motifs[1]
        self.batch_nodes_inds: torch.Tensor = meta_info['x']
        self.batch_y_inds: torch.Tensor = meta_info['y']
        self.batch_edges_inds: torch.Tensor = meta_info['edge_index']


        if self.dataspec == DataSpec.TRAIN:
            l = 0.8 * dataset_num
            self.graph_inds = torch.arange(l)

        elif self.dataspec == DataSpec.VAL:
            l1 = 0.8 * dataset_num
            l2 = 0.9 * dataset_num
            self.graph_inds = torch.arange(l1, l2)
        
        else:
            l = 0.9 * dataset_num
            self.graph_inds = torch.arange(l, dataset_num)

    
    def __len__(self):
        return self.graph_inds.numel()

    def __getitem__(self, inds):
        datas: List[Data] = []
        
        if isinstance(inds, int):
            inds = [inds]

        for idx in inds:
            graph_ind = self.graph_inds[idx].long()
            batch_node_ind_start = self.batch_nodes_inds[graph_ind]
            batch_node_ind_end = self.batch_nodes_inds[graph_ind + 1]
            x = self.data.x[batch_node_ind_start: batch_node_ind_end]

            batch_edge_ind_start = self.batch_edges_inds[graph_ind]
            batch_edge_ind_end = self.batch_edges_inds[graph_ind + 1]
            edge_index = self.data.edge_index[:, batch_edge_ind_start: batch_edge_ind_end]            

            batch_y_ind_start = self.batch_y_inds[graph_ind]
            batch_y_ind_end = self.batch_y_inds[graph_ind + 1]
            y = self.data.y[batch_y_ind_start: batch_y_ind_end]

            data = Data(x=x, edge_index=edge_index, y=y)
            datas.append(data)
        
        return datas