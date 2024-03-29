
from dig.xgraph.models import *
import torch

from .....config import BaseConfig, TrainingConfig
from .....enums import *
from ....core import TrainEntrypoint

class Entrypoint(TrainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=12,
            try_name='gin3l',
            dataset_name=Dataset.MUTAG,
            training_config=TrainingConfig(500, OptimType.ADAM, batch_size=32, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
        )
        
        model = GIN_3l(model_level='graph', dim_node=7, dim_hidden=60, num_classes=2)
                
        super(Entrypoint, self).__init__(conf, model)