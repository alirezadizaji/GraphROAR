from argparse import ArgumentParser

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, DataLoader

from GNN_Explainability.entrypoints.core.main import get_dataset_cls
from GNN_Explainability.enums import Dataset, DataSpec
from GNN_Explainability.utils.symmetric_edge_mask import symmetric_edges

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-K', '--keep', action='store_true', help='If given then keep most informative parts.')
    parser.add_argument('-N', '--normalize', action='store_true', help='If given then normalize.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    cls = get_dataset_cls(Dataset.BA3Motifs)
    
    train_set = cls(DataSpec.TRAIN)
    val_set = cls(DataSpec.VAL)
    test_set = cls(DataSpec.TEST)

    val = DataLoader(val_set, batch_size=1)

    # model = GCN_3l_BN(model_level='graph', dim_node=10, dim_hidden=20, num_classes=2)
    # model = GIN_3l(model_level='graph', dim_node=7, dim_hidden=60, num_classes=2)
    model = GCN_3l_BN(model_level='graph', dim_node=1, dim_hidden=20, num_classes=3)
    model.to('cuda:0')
    # model.load_state_dict(torch.load('../results/1_gcn3l_BA2Motifs/weights/16', map_location='cuda:0'))
    # model.load_state_dict(torch.load('../results/12_gin3l_MUTAG/weights/176', map_location='cuda:0'))
    model.load_state_dict(torch.load('../results/229_gcn3l_BA3Motifs/weights/61', map_location='cuda:0'))
    model.eval()

    N = 0
    fidelity = 0.0
    update_N = True
    
    sparsities = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    fidelities = np.zeros_like(sparsities, dtype=np.float32)
    method = 'subgraphx_90%'
    for data in val:
        data = data[0]
        data = data.to('cuda:0')
        num_edges = data.edge_index.size(1)
        pth = f"../data/BA3Motifs/explanation/gcn3l/{method}/{data.name[0]}.npy"
        mask = torch.from_numpy(np.load(pth))
        mask = symmetric_edges(data.edge_index, mask)
        out = model(data=data).squeeze().sigmoid()
        # out = F.softmax(out)
        c = out.argmax()
        p1 = out[c]
        
        N += 1

        for ix, sparsity in enumerate(sparsities):
            g = data.clone()

            k = int((1 - sparsity) * num_edges)
            # print(k)

            _, indices = mask.topk(k)

            if args.keep:
                masskk = torch.zeros_like(g.edge_index[0]).bool()
                masskk[indices] = True
            else:
                masskk = torch.ones_like(g.edge_index[0]).bool()
                masskk[indices] = False

            g.edge_index = g.edge_index[:, masskk]
            
            B = g.x.size(0)
            remained: torch.Tensor = torch.unique(g.edge_index)
            node_mask = torch.zeros(B).bool()
            node_mask[remained] = True
            g.x[~node_mask] = 0

            # # let's nodes indices start from zero
            # row, col = g.edge_index
            # node_indices, _ = torch.sort(torch.unique(g.edge_index))
            # mapping = torch.full((node_indices.max().item() + 1,), fill_value=torch.inf)
            # mapping[node_indices] = torch.arange(node_indices.numel()).float()
            
            # masked_edges = torch.stack([mapping[row], mapping[col]], dim=0).long()
            # masked_edges = masked_edges.to(g.edge_index.device)
            # g.edge_index = masked_edges

            # delattr(g, 'ptr')
            # delattr(g, 'batch')
            # g: Batch = Batch.from_data_list([g])
            outt = model(data=g).squeeze().sigmoid()
            # out = F.softmax(out)
            p2 = outt[c]
            # print(sparsity, g.edge_index.shape, g.x.shape, p1.item(), p2.item(), g.y.item(), c.item())
            diff = p1.item() - p2.item()
            if args.normalize:
                diff = diff / p1.item()

            fidelities[ix] += diff
            # if g.y.item() == 1:
            #     print(p1.item(), p2.item(), c.item(), g.y.item(), g.edge_index.shape)
        
    fidelities /= N 
    print(fidelities)
