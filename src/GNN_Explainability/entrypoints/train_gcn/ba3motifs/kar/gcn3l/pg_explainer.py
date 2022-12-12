from tqdm import tqdm

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, Batch


from ......config import RetrainingConfig, TrainingConfig
from ......enums import *
from .....core import ROAREntrypoint

class Entrypoint(ROAREntrypoint):
    
    def __init__(self):
        conf = RetrainingConfig(
            try_num=262,
            try_name='kar_pgexplainer_gcn3l',
            dataset_name=Dataset.BA3Motifs,
            training_config=TrainingConfig(100, OptimType.ADAM),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=f'../data/{Dataset.BA3Motifs}/explanation/gcn3l/pgexplainer',
            eliminate_top_most_edges=False,
            retraining_ratiosios=[0.1, 0.3, 0.5, 0.7, 0.9],
        )

        model = GCN_3l_BN(model_level='graph', dim_node=1, dim_hidden=20, num_classes=3)

        super(Entrypoint, self).__init__(conf, model)