from tqdm import tqdm

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, Batch


from ......config import ROARConfig, TrainingConfig
from ......enums import *
from .....core import ROAREntrypoint

class Entrypoint(ROAREntrypoint):
    
    def __init__(self):
        conf = ROARConfig(
            try_num=496,
            try_name='kar_random_gin3l_skip_eval',
            dataset_name=Dataset.MSRC9,
            training_config=TrainingConfig(500, OptimType.ADAM, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=f'../data/{Dataset.MSRC9}/explanation/gin3l/random',
            edge_mask_random_weighting=True, # random roar
            eliminate_top_most_edges=False,
            roar_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
            skip_during_evaluation=True,
        )

        model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=20, num_classes=8)

        super(Entrypoint, self).__init__(conf, model)