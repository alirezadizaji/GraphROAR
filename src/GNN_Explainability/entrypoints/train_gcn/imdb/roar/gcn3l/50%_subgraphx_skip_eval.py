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
            try_num=446,
            try_name='roar_subgraphx_0.5_gcn3l_skip_eval',
            dataset_name=Dataset.IMDB_BIN,
            training_config=TrainingConfig(500, OptimType.ADAM, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=f'../data/{Dataset.IMDB_BIN}/explanation/gcn3l/subgraphx_50%',
            retraining_ratiosios=[0.5],
            skip_during_evaluation=True,
        )

        model = GCN_3l_BN(model_level='graph', dim_node=1, dim_hidden=20, num_classes=2)

        super(Entrypoint, self).__init__(conf, model)