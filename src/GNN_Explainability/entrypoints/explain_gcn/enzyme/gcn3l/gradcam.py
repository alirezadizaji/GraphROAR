import os

from dig.xgraph.method import GradCAM
from dig.xgraph.models import *
import numpy as np
import torch

from dig.xgraph.models import *
import numpy as np
import torch

from GNN_Explainability.config.base_config import TrainingConfig

from .....entrypoints.core import GradCAMEntrypoint

from .....config import ExplainConfig
from .....enums import *
from .....utils.symmetric_edge_mask import symmetric_edges


class Entrypoint(GradCAMEntrypoint):
    def __init__(self):
        conf = ExplainConfig(
            try_num=286,
            try_name='gradcam_gcn3l',
            dataset_name=Dataset.ENZYME,
            training_config=TrainingConfig(300, OptimType.ADAM),
            save_log_in_file=True,
            num_classes=6,
            save_visualization=True,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.ENZYME, 'explanation', 'gcn3l', 'gradcam'),
            num_instances_to_visualize=20,
            sparsity=0.0,
            node_color_setter=None,
            plt_legend=None,
            explain_graph=True,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        )
        
        model = GCN_3l_BN(model_level='graph', dim_node=18, dim_hidden=60, num_classes=conf.num_classes)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/283_gcn3l_ENZYMES/weights/234', map_location=conf.device))

        explainer = GradCAM(model, explain_graph=conf.explain_graph)

        super(Entrypoint, self).__init__(conf, model, explainer)