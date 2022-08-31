from tqdm import tqdm

from dig.xgraph.models import *
import torch


from ......config import ROARConfig, TrainingConfig
from ......enums import *
from .....core import ROAREntrypoint

class Entrypoint(ROAREntrypoint):
    
    def __init__(self):
        conf = ROARConfig(
            try_num=58,
            try_name='kar_gnnexplainer_gin3l_skip_eval',
            dataset_name=Dataset.BA2Motifs,
            training_config=TrainingConfig(100, OptimType.ADAM),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=False,
            edge_masks_load_dir='../data/ba_2motifs/explanation/gnnexplainer',
            eliminate_top_most_edges=False,
            roar_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
            skip_during_evaluation=True,
        )

        model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)

        super(Entrypoint, self).__init__(conf, model)