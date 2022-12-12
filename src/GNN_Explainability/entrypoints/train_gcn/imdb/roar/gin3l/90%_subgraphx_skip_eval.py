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
            try_num=457,
            try_name='roar_subgraphx_0.9_gin3l_skip_eval',
            dataset_name=Dataset.IMDB_BIN,
            training_config=TrainingConfig(500, OptimType.ADAM, early_stop=100),
            device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=f'../data/{Dataset.IMDB_BIN}/explanation/gin3l/subgraphx_90%',
            retraining_ratiosios=[0.9],
            skip_during_evaluation=True,
        )

        model = GIN_3l(model_level='graph', dim_node=1, dim_hidden=20, num_classes=2)

        super(Entrypoint, self).__init__(conf, model)