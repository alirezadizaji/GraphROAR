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
            try_num=305,
            try_name='kar_subgraphx_0.5_gcn3l_skip_eval',
            dataset_name=Dataset.ENZYME,
            training_config=TrainingConfig(500, OptimType.ADAM, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=f'../data/{Dataset.ENZYME}/explanation/gcn3l/subgraphx_50%',
            roar_ratios=[0.5],
            skip_during_evaluation=True,
            eliminate_top_most_edges=False
        )

        model = GCN_3l_BN(model_level='graph', dim_node=18, dim_hidden=60, num_classes=6)

        super(Entrypoint, self).__init__(conf, model)