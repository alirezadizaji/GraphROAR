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
            try_num=500,
            try_name='roar_subgraphx_0.7_gcn3l_skip_eval',
            dataset_name=Dataset.MSRC9,
            training_config=TrainingConfig(500, OptimType.ADAM, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=f'../data/{Dataset.MSRC9}/explanation/gcn3l/subgraphx_70%',
            roar_ratios=[0.7],
            skip_during_evaluation=True,
        )

        model = GCN_3l_BN(model_level='graph', dim_node=10, dim_hidden=20, num_classes=8)

        super(Entrypoint, self).__init__(conf, model)