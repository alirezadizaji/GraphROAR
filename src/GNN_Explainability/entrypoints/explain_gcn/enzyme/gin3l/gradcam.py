import os

from dig.xgraph.method import GradCAM
from dig.xgraph.models import *
import numpy as np
import torch

from dig.xgraph.models import *
import numpy as np
import torch

from .....config.base_config import TrainingConfig

from .....entrypoints.core import GradCAMEntrypoint

from .....config import ExplainConfig
from .....enums import *
from .....utils.symmetric_edge_mask import symmetric_edges


class Entrypoint(GradCAMEntrypoint):
    def __init__(self):
        conf = ExplainConfig(
            try_num=294,
            try_name='gradcam',
            dataset_name=Dataset.ENZYME,
            training_config=TrainingConfig(300, OptimType.ADAM),
            save_log_in_file=True,
            num_classes=6,
            save_visualization=True,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.ENZYME, 'explanation', 'gin3l', 'gradcam'),
            num_instances_to_visualize=20,
            sparsity=0.0,
            node_color_setter=None,
            plt_legend=None,
            explain_graph=True,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        )
        
        model = GIN_3l(model_level='graph', dim_node=18, dim_hidden=80, num_classes=conf.num_classes)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/284_gin3l_ENZYMES/weights/404', map_location=conf.device))

        explainer = GradCAM(model, explain_graph=conf.explain_graph)

        super(Entrypoint, self).__init__(conf, model, explainer)

    def _select_explainable_edges(self, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        edge_mask = symmetric_edges(edge_index, edge_mask)
        k = 12
        edge_index = edge_index[:, edge_mask.topk(k)[1]]

        return edge_index