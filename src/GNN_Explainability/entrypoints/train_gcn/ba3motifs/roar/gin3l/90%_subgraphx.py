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
            try_num=382,
            try_name='roar_subgraphx_0.9_gin3l',
            dataset_name=Dataset.BA3Motifs,
            training_config=TrainingConfig(100, OptimType.ADAM),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=f'../data/{Dataset.BA3Motifs}/explanation/gin3l/subgraphx_90%',
            roar_ratios=[0.9],
        )

        model = GIN_3l(model_level='graph', dim_node=1, dim_hidden=20, num_classes=3)

        super(Entrypoint, self).__init__(conf, model)