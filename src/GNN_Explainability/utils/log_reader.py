from argparse import ArgumentParser
from importlib import import_module
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from GNN_Explainability.entrypoints.core.main import MainEntrypoint

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-T', '--type', choices=['norm', 'skip'], default='skip', help='Entrypoint types: non-skip or skip evaluation.')
    parser.add_argument('-D', '--dir', type=str, help='Directory to take entrypoints')
    parser.add_argument('-GR', '--gradcam', type=str, default='gradcam', help='gradcam entry name')
    parser.add_argument('-GX', '--gnnexplainer', type=str, default='gnnexplainer', help='gnnexplainer entry name')
    parser.add_argument('-PX', '--pgexplainer', type=str, default='pgexplainer', help='pgexplainer entry name.')
    parser.add_argument('-SX', '--subgraphx', type=str, default='subgraphx', help='subgraphx entry name.')
    parser.add_argument('-RN', '--random', type=str, default='random', help='Random entry name.')
    parser.add_argument('-N', '--count', type=int, help='Number of consoles to read.')
    parser.add_argument('-H', '--high', type=float, default=100, help='High percentage to consider.')
    parser.add_argument('-L', '--low', type=float, default=50, help='Low percentage to consider.')
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    suffix = "_skip_eval" if args.type == "skip" else ""
    res = {}
    dirs = {}
    print(f"@@@ Getting {args.dir} entries with {args.count} logs each for {args.type} running type @@@", flush=True)
    for X in ['gradcam', 'gnnexplainer', 'pgexplainer', 'subgraphx', 'random']:
        if X == 'subgraphx':
            ps = [10, 30, 50, 70, 90]
            res.setdefault(X, [[] for _ in range(args.count)])
            for p in ps:

                entry_name = f"{p}%_{getattr(args, X)}{suffix}"
                script = import_module(f"GNN_Explainability.entrypoints.{args.dir}.{entry_name}")
                entrypoint: 'MainEntrypoint' = getattr(script, 'Entrypoint')()
                save_dir = entrypoint.conf.save_dir
                dirs.setdefault(X, [])
                dirs[X].append(save_dir)
                files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f)) and f.endswith("_o.log")]
                files = list(sorted(files, reverse=True))
                
                for c, f in enumerate(files[:args.count]):
                    pth = os.path.join(save_dir, f)
                
                    with open(pth, 'r') as f:
                        lines = f.readlines()
                
                        for i, l in enumerate(lines):
                            if l.startswith("@@@@@"):
                                acc = lines[i+1].split(",")[0].replace("val: ", "").replace(" ", "")
                                acc = round(float(acc) * 100, 2)
                
                                res[X][c].append(acc)
                                break
        else:
            entry_name = f"{getattr(args, X)}{suffix}"
            script = import_module(f"GNN_Explainability.entrypoints.{args.dir}.{entry_name}")
            entrypoint: 'MainEntrypoint' = getattr(script, 'Entrypoint')()
            save_dir = entrypoint.conf.save_dir
            dirs[X] = save_dir

            files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f)) and f.endswith("_o.log")]
            files = list(sorted(files, reverse=True))
            
            for c, f in enumerate(files[:args.count]):
                pth = os.path.join(save_dir, f)
            
                with open(pth, 'r') as f:
                    lines = f.readlines()
            
                    for i, l in enumerate(lines):
                        if l.startswith("@@@@@ B"):
                            acc = lines[i+1].split(",")[0].replace("val: ", "").replace(" ", "")
                            acc = round(float(acc) * 100, 2)

                            res.setdefault(X, [[] for _ in range(args.count)])
                            res[X][c].append(acc)

    for k, v in res.items():
        for l in v:
            l.insert(0, args.low)
            l.append(args.high)
        res[k] = v
        print(f"{k} = {v}")
    print()
    for k, v in res.items():
        for i, l in enumerate(v):
            if len(l) != 7:
                print(f"WARNING -> {k} {i}th")

    print()
    print(dirs)


if __name__ == "__main__":
    main()