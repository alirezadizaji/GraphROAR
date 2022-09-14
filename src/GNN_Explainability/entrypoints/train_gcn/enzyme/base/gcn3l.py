
from dig.xgraph.models import *
import torch

from .....config import BaseConfig, TrainingConfig
from .....enums import *
from ....core import TrainEntrypoint

class Entrypoint(TrainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=283,
            try_name='gcn3l',
            dataset_name=Dataset.ENZYME,
            training_config=TrainingConfig(500, OptimType.ADAM, lr=1e-3, batch_size=64, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
        )
        
        model = GCN_3l_BN(model_level='graph', dim_node=18, dim_hidden=60, num_classes=6)
                
        super(Entrypoint, self).__init__(conf, model)