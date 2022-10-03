import os

from dig.xgraph.method import GNNExplainer
from dig.xgraph.models import *
import torch

from dig.xgraph.models import *
import numpy as np
import torch

from .....config.base_config import TrainingConfig

from .....entrypoints.core import GNNExplainerEntrypoint
from .....utils.symmetric_edge_mask import symmetric_edges

from .....config import GNNExplainerConfig
from .....enums import *

def color_setter(x):
    i = torch.nonzero(x).item()
    if i == 0:   # Void
        return Color.BLACK
    elif i == 1: # Building
        return Color.MAROON
    elif i == 2: # Grass
        return Color.GREEN_V2
    elif i == 3: # Tree
        return Color.OlIVE
    elif i == 4: # Cow
        return Color.NAVY
    elif i == 5: # Sky
        return Color.RED_V2
    elif i == 6: # Aeroplane
        return Color.GREY
    elif i == 7: # Face
        return Color.DARK_YELLOW
    elif i == 8: # Car
        return Color.PURPLE
    elif i == 9: # Bicycle
        return Color.PINK_V2



legend = {Color.BLACK: "Void", 
    Color.MAROON: "Building",
    Color.GREEN_V2: "Grass",
    Color.OlIVE: "Tree",
    Color.NAVY: "Cow", 
    Color.RED_V2: "Sky",
    Color.GREY: "Aeroplane",
    Color.DARK_YELLOW: "Face",
    Color.PURPLE: "Car",
    Color.PINK_V2: "Bicycle"}

class Entrypoint(GNNExplainerEntrypoint):
    def __init__(self):
        conf = GNNExplainerConfig(
            try_num=464,
            try_name='gnnexplainer',
            dataset_name=Dataset.MSRC9,
            training_config=TrainingConfig(500, OptimType.ADAM, 0.01, batch_size=1),
            save_log_in_file=True,
            num_classes=8,
            save_visualization=True,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.MSRC9, 'explanation', 'gin3l', 'gnnexplainer'),
            num_instances_to_visualize=50,
            explain_graph=True,
            mask_features=True,
            coff_edge_size=0.0,
            sparsity=0.0,
            node_color_setter=color_setter,
            plt_legend=legend,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            coff_node_feat_size=0.0)
        
        model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=20, num_classes=8)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/463_gin3l_MSRC_9/weights/53', map_location=conf.device))
        
        explainer = GNNExplainer(model, epochs=conf.training_config.num_epochs,
                 lr=conf.training_config.lr, explain_graph=conf.explain_graph, 
                 coff_edge_size=conf.coff_edge_size, coff_node_feat_size=conf.coff_node_feat_size)

        super().__init__(conf, model, explainer)

    def _select_explainable_edges(self, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        edge_mask = symmetric_edges(edge_index, edge_mask)
        k = int(edge_mask.numel() * 0.3)
        edge_index = edge_index[:, edge_mask.topk(k)[1]]

        return edge_index