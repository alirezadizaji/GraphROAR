import os

from dig.xgraph.models import *
import torch


from ......config import ROARConfig, TrainingConfig
from ......enums import *
from .....core import ROAREntrypoint

class Entrypoint(ROAREntrypoint):
    
    def __init__(self):
        conf = ROARConfig(
            try_num=228,
            try_name='kar_random_gin3l',
            dataset_name=Dataset.REDDIT_BINARY,
            training_config=TrainingConfig(500, OptimType.ADAM, batch_size=32, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=os.path.join('..', 'data', Dataset.REDDIT_BINARY, 'explanation', 'gin3l', 'random'),
            edge_mask_random_weighting=True, # random roar
            eliminate_top_most_edges=False,
            roar_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        )

        model = GIN_3l(model_level='graph', dim_node=1, dim_hidden=60, num_classes=2)

        super(Entrypoint, self).__init__(conf, model)