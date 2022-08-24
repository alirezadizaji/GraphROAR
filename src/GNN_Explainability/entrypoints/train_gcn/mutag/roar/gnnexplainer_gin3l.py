from tqdm import tqdm

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, Batch


from .....config import ROARConfig, TrainingConfig
from .....enums import *
from ....core import ROAREntrypoint

class Entrypoint(ROAREntrypoint):
    
    def __init__(self):
        conf = ROARConfig(
            try_num=17,
            try_name='roar_gnnexplainer_gin3l',
            dataset_name=Dataset.MUTAG,
            training_config=TrainingConfig(500, OptimType.ADAM, batch_size=128),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir='../data/MUTAG/explanation/gnnexplainer',
            roar_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        )

        model = GIN_3l(model_level='graph', dim_node=14, dim_hidden=300, num_classes=2)

        super(Entrypoint, self).__init__(conf, model)