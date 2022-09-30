import os

from dig.xgraph.method import GradCAM
from dig.xgraph.models import *
import torch

from dig.xgraph.models import *
import torch

from .gnn_explainer import color_setter, legend

from .....entrypoints.core import GradCAMEntrypoint

from .....config import ExplainConfig, TrainingConfig
from .....enums import *


class Entrypoint(GradCAMEntrypoint):
    def __init__(self):
        conf = ExplainConfig(
            try_num=465,
            try_name='gradcam',
            dataset_name=Dataset.MSRC9,
            training_config=TrainingConfig(300, OptimType.ADAM),
            save_log_in_file=True,
            num_classes=8,
            save_visualization=True,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.MSRC9, 'explanation', 'gin3l', 'gradcam'),
            num_instances_to_visualize=50,
            sparsity=0.0,
            node_color_setter=color_setter,
            plt_legend=legend,
            explain_graph=True,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        )
        
        model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=20, num_classes=8)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/463_gin3l_MSRC_9/weights/53', map_location=conf.device))
 
        explainer = GradCAM(model, explain_graph=conf.explain_graph)

        super(Entrypoint, self).__init__(conf, model, explainer)