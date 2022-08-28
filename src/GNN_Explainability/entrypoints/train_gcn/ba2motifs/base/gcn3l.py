
from dig.xgraph.models import *
import torch

from .....config import BaseConfig, TrainingConfig
from .....enums import *
from ....core import TrainEntrypoint

class Entrypoint(TrainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=1,
            try_name='gcn3l',
            dataset_name=Dataset.BA2Motifs,
            training_config=TrainingConfig(100, OptimType.ADAM, lr=1e-3, batch_size=64),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
        )
        
        model = GCN_3l_BN(model_level='graph', dim_node=10, dim_hidden=20, num_classes=2)
                
        super(Entrypoint, self).__init__(conf, model)