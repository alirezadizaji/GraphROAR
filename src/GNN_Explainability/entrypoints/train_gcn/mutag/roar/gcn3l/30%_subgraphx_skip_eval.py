import os

from dig.xgraph.models import *
import torch


from ......config import RetrainingConfig, TrainingConfig
from ......enums import *
from .....core import RetrainingEntrypoint

class Entrypoint(RetrainingEntrypoint):
    
    def __init__(self):
        conf = RetrainingConfig(
            try_num=166,
            try_name='roar_subgraphx_0.3_gcn3l_skip_eval',
            dataset_name=Dataset.MUTAG,
            training_config=TrainingConfig(500, OptimType.ADAM, batch_size=32, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=os.path.join('..', 'data', 'MUTAG', 'explanation', 'gcn3l', 'subgraphx_30%'),
            retraining_ratiosios=[0.3],
            skip_during_evaluation=True,
        )

        model = GCN_3l_BN(model_level='graph', dim_node=7, dim_hidden=60, num_classes=2)

        super(Entrypoint, self).__init__(conf, model)