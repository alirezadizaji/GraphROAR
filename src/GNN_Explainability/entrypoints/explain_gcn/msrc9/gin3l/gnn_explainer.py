import os

from dig.xgraph.method import GNNExplainer
from dig.xgraph.models import *
import torch

from dig.xgraph.models import *
import numpy as np
import torch

from .....config.base_config import TrainingConfig

from .....entrypoints.core import GNNExplainerEntrypoint

from .....config import GNNExplainerConfig
from .....enums import *

def color_setter(x):
    i = torch.nonzero(x).item()
    if i == 0:   # Void
        return Color.BLACK
    elif i == 1: # Building
        return Color.MAROON
    elif i == 1: # Grass
        return Color.GREEN_V2
    elif i == 1: # Tree
        return Color.OlIVE
    elif i == 1: # Cow
        return Color.NAVY
    elif i == 1: # Aeroplane
        return Color.RED_V2
    elif i == 1: # Sky
        return Color.GREY
    elif i == 1: # Face
        return Color.DARK_YELLOW
    elif i == 1: # Car
        return Color.PURPLE
    elif i == 1: # Bicycle
        return Color.PINK_V2



legend = {Color.BLACK: "Void", 
    Color.MAROON: "Building",
    Color.GREEN: "Grass",
    Color.OlIVE: "Tree",
    Color.NAVY: "Cow", 
    Color.RED_V2: "Aeroplane",
    Color.GREY: "Sky",
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