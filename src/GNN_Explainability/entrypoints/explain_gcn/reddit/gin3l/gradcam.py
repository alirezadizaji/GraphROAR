import os

from dig.xgraph.method import GradCAM
from dig.xgraph.models import *
import torch

from dig.xgraph.models import *
import torch

from .....entrypoints.core import GradCAMEntrypoint

from .....config import ExplainConfig, TrainingConfig
from .....enums import *


class Entrypoint(GradCAMEntrypoint):
    def __init__(self):
        conf = ExplainConfig(
            try_num=190,
            try_name='gradcam',
            dataset_name=Dataset.REDDIT_BINARY,
            training_config=TrainingConfig(300, OptimType.ADAM),
            save_log_in_file=True,
            num_classes=2,
            save_visualization=True,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.REDDIT_BINARY, 'explanation', 'gin3l', 'gradcam'),
            num_instances_to_visualize=20,
            sparsity=0.0,
            explain_graph=True,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            plt_legend=None,
            node_color_setter=None)
        
        model = GIN_3l(model_level='graph', dim_node=1, dim_hidden=60, num_classes=conf.num_classes)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/180_gin3l_REDDIT-BINARY/weights/208', map_location=conf.device))

        explainer = GradCAM(model, explain_graph=conf.explain_graph)

        super(Entrypoint, self).__init__(conf, model, explainer)
    
    def _select_explainable_edges(self, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        k = int(edge_mask.numel() * 0.5)
        edge_index = edge_index[:, edge_mask.topk(k)[1]]

        return edge_index