
from dig.xgraph.models import *
import torch

from .....config import BaseConfig, TrainingConfig
from .....enums import *
from ....core import TrainEntrypoint

class Entrypoint(TrainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=230,
            try_name='gin3l',
            dataset_name=Dataset.BA3Motifs,
            training_config=TrainingConfig(100, OptimType.ADAM, batch_size=64),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
        )
        
        model = GIN_3l(model_level='graph', dim_node=1, dim_hidden=20, num_classes=3)
                
        super(Entrypoint, self).__init__(conf, model)