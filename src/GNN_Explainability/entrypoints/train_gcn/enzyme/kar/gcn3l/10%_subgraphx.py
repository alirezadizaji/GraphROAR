from tqdm import tqdm

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, Batch


from ......config import RetrainingConfig, TrainingConfig
from ......enums import *
from .....core import RetrainingEntrypoint

class Entrypoint(RetrainingEntrypoint):
    
    def __init__(self):
        conf = RetrainingConfig(
            try_num=302,
            try_name='kar_subgraphx_0.1_gcn3l',
            dataset_name=Dataset.ENZYME,
            training_config=TrainingConfig(500, OptimType.ADAM, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=f'../data/{Dataset.ENZYME}/explanation/gcn3l/subgraphx_10%',
            retraining_ratiosios=[0.1],
            eliminate_top_most_edges=False,
            
        )

        model = GCN_3l_BN(model_level='graph', dim_node=18, dim_hidden=80, num_classes=6)

        super(Entrypoint, self).__init__(conf, model)