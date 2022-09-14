
from dig.xgraph.models import *
import torch

from .....config import BaseConfig, TrainingConfig
from .....enums import *
from ....core import TrainEntrypoint

class Entrypoint(TrainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=284,
            try_name='gin3l',
            dataset_name=Dataset.ENZYME,
            training_config=TrainingConfig(500, OptimType.ADAM, batch_size=64, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
        )
        
        model = GIN_3l(model_level='graph', dim_node=18, dim_hidden=80, num_classes=6)
                
        super(Entrypoint, self).__init__(conf, model)