from argparse import ArgumentParser
from importlib import import_module
import os
import random
from typing import Dict, Tuple

from dig.xgraph.models import *
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, DataLoader

from ..entrypoints.core.main import get_dataset_cls
from ..entrypoints.core.train import TrainEntrypoint
from ..enums import Dataset, DataSpec
from .symmetric_edge_mask import symmetric_edges

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-G', '--globale', type=int, default=12345, help='seed.')
    parser.add_argument('-K', '--keep', action='store_true', help='If given then keep most informative parts.')
    parser.add_argument('-N', '--normalize', action='store_true', help='If given then normalize.')
    parser.add_argument('-E', '--entrypoint', type=str, help='Directory to look for the Entrypoint class (Type=train_gcn). The class contains the model and the configuration of running.')
    parser.add_argument('-C', '--epoch_num', type=int, help='Epoch number to be loaded. Please checkout corresponding entrypoint.')
    parser.add_argument('-X', '--edge_mask_dir', nargs="+", type=str, help='Directories, representing edge masks of methods.')
    parser.add_argument('-O', '--color', nargs="+", type=str, help='Colors to be depicted per method.')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    seed = args.globale
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    script = import_module(args.entrypoint)
    entrypoint: 'TrainEntrypoint' = getattr(script, 'Entrypoint')()
    cls = get_dataset_cls(entrypoint.conf.dataset_name)

    model: nn.Module = entrypoint.model
    model.load_state_dict(torch.load(entrypoint._get_weight_save_path(args.epoch_num), map_location=entrypoint.conf.device))
    model.to(entrypoint.conf.device)
    model.eval()

    val_set = cls(DataSpec.VAL)
    val_loader = DataLoader(val_set, batch_size=1)
    num_samples = val_loader.dataset.__len__()

    fidelities: Dict[str, np.ndarray] = dict()
    sparsities: Dict[str, np.ndarray] = dict()
    subgraphx_exist: bool = False

    colors: Dict[str, str] = dict()

    for c, d in zip(args.color, args.edge_mask_dir):
        method: str = os.path.basename(d)
        method = method.lower()

        colors[method] = c

        # Let's determine sparsity interval
        ## All methods, except for subgraphx, has static interval of [0.0, 1.0]
        if "subgraphx" not in method:
            start = 0.0
            end = 1.0
        ## Subgraphx have the same interval but it must be performed on multiple sub-intervals individually.
        else:
            subgraphx_exist = True
            import re
            p = int(re.findall(r'\d+', method)[0])
            if p == 10:
                start = 0.0
                end = 0.1
            elif p == 90:
                start = 0.7
                end = 1.0
            else:
                end = p / 100
                start = end - 0.2

        num = (end - start) / 0.01
        sparsities[method] = np.linspace(start, end, num=int(num))
        fidelities[method] = np.zeros_like(sparsities[method], dtype=np.float32)

        for i, data in enumerate(val_loader):
            data: Batch = data[0]
            data = data.to(entrypoint.conf.device)
            
            out = model(data=data).squeeze().sigmoid()
            c = out.argmax()
            p1 = out[c]
            
            pth = os.path.join(d, f"{data.name[0]}.npy")
            mask = torch.from_numpy(np.load(pth))
            mask = symmetric_edges(data.edge_index, mask)

            num_edges = data.edge_index.size(1)
            
            for ix, sparsity in enumerate(sparsities[method]):
                g = data.clone()

                k = int((sparsity) * num_edges)

                _, indices = mask.topk(k)

                if args.keep:
                    masskk = torch.zeros_like(g.edge_index[0]).bool()
                    masskk[indices] = True
                else:
                    masskk = torch.ones_like(g.edge_index[0]).bool()
                    masskk[indices] = False

                g.edge_index = g.edge_index[:, masskk]
                
                if g.edge_index.numel() == 0:
                    g.x = torch.zeros_like(g.x)
                ## node indices must be reset
                else:
                    B = g.x.size(0)
                    node_mask = torch.zeros(B).bool()
                    remained: torch.Tensor = torch.unique(g.edge_index)
                    node_mask[remained] = True
                    g.x = g.x[node_mask]

                    row, col = g.edge_index
                    node_indices, _ = torch.sort(torch.unique(g.edge_index))
                    mapping = torch.full((node_indices.max().item() + 1,), fill_value=torch.inf)
                    mapping[node_indices] = torch.arange(node_indices.numel()).float()
                    
                    masked_edges = torch.stack([mapping[row], mapping[col]], dim=0).long()
                    g.edge_index = masked_edges.to(g.edge_index.device)

                delattr(g, 'ptr')
                delattr(g, 'batch')
                g: Batch = Batch.from_data_list([g])
                outt = model(data=g).squeeze().sigmoid()
                p2 = outt[c]

                diff = p1.item() - p2.item()
                if args.normalize:
                    diff = diff / p1.item()

                fidelities[method][ix] += diff
        
        fidelities[method] /= num_samples 
        print(f"DONE!!! Method: {method}, Keep: {args.keep}, Normalized: {args.normalize}", flush=True)
        fidelities[method] = np.vectorize(lambda x: round(x*100, 4))(fidelities[method])

    # Join subgraphx fidelities into a single fidelity
    if subgraphx_exist:
        subgraphx_info: Dict[float, Tuple[np.ndarray, np.ndarray]] = dict()
        methods = list(fidelities.keys())
        for k in methods:
            if "subgraphx" not in k:
                continue
            
            s = sparsities[k]
            f = fidelities[k]
            max_bound = s.max()
            subgraphx_info[max_bound] = (s, f)

            del fidelities[k]
            del sparsities[k]

        subgraphx_info = dict(sorted(subgraphx_info.items()))
        colors["subgraphx"] = "DB0000"
        sparsities["subgraphx"] = np.concatenate([v[0] for v in subgraphx_info.values()])
        fidelities["subgraphx"] = np.concatenate([v[1] for v in subgraphx_info.values()])

    sparsity_type = "KSparsity" if args.keep else "RSparsity"

    import matplotlib.pyplot as plt

    plt.xlabel(f"{sparsity_type} (%)")
    plt.ylabel("Fidelity (%)")

    for method, fidelity in fidelities.items():
        sparsity = sparsities[method]
        color = colors[method]
        plt.plot(sparsity, fidelity)
    
    methods = list(fidelities.keys())
    plt.legend(methods)
    plt.savefig("salam.png")