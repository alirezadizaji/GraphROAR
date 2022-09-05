import os

from dig.xgraph.models import *
import torch


from ......config import ROARConfig, TrainingConfig
from ......enums import *
from .....core import ROAREntrypoint

class Entrypoint(ROAREntrypoint):
    
    def __init__(self):
        conf = ROARConfig(
            try_num=147,
            try_name='kar_subgraphx_0.1_gcn3l_skip_eval',
            dataset_name=Dataset.MUTAG,
            training_config=TrainingConfig(500, OptimType.ADAM, batch_size=32, early_stop=100),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            edge_masks_load_dir=os.path.join('..', 'data', 'MUTAG', 'explanation', 'gcn3l', 'subgraphx_10%'),
            eliminate_top_most_edges=False,
            roar_ratios=[0.1],
            skip_during_evaluation=True,
        )

        model = GCN_3l_BN(model_level='graph', dim_node=7, dim_hidden=60, num_classes=2)

        super(Entrypoint, self).__init__(conf, model)