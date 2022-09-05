import os

from dig.xgraph.models import *
import torch


from ......config import ROARConfig, TrainingConfig
from ......enums import *
from .....core import ROAREntrypoint

class Entrypoint(ROAREntrypoint):
    
    def __init__(self):
        conf = ROARConfig(
            try_num=158,
            try_name='kar_subgraphx_0.7_gin3l_skip_eval',
            dataset_name=Dataset.MUTAG,
            training_config=TrainingConfig(500, OptimType.ADAM, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=os.path.join('..', 'data', 'MUTAG', 'explanation', 'gin3l', 'subgraphx_70%'),
            eliminate_top_most_edges=False,
            roar_ratios=[0.7],
            skip_during_evaluation=True,
        )

        model = GIN_3l(model_level='graph', dim_node=7, dim_hidden=60, num_classes=2)

        super(Entrypoint, self).__init__(conf, model)