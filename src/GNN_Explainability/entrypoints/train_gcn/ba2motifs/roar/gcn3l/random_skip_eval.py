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
            try_num=78,
            try_name='roar_random_gcn3l_skip_eval',
            dataset_name=Dataset.BA2Motifs,
            training_config=TrainingConfig(100, OptimType.ADAM),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir='../data/ba_2motifs/explanation/gcn3l/random',
            edge_mask_random_weighting=True, # random roar
            retraining_ratiosios=[0.1, 0.3, 0.5, 0.7, 0.9],
            skip_during_evaluation=True,
        )

        model = GCN_3l_BN(model_level='graph', dim_node=10, dim_hidden=20, num_classes=2)

        super(Entrypoint, self).__init__(conf, model)