import os

from dig.xgraph.method import GNNExplainer
from dig.xgraph.models import *
import numpy as np
import torch

from dig.xgraph.models import *
import numpy as np
import torch

from GNN_Explainability.config.base_config import TrainingConfig

from .....entrypoints.core import GNNExplainerEntrypoint

from .....config import GNNExplainerConfig
from .....enums import *


class Entrypoint(GNNExplainerEntrypoint):
    def __init__(self):
        conf = GNNExplainerConfig(
            try_num=231,
            try_name='gnnexplainer_gcn3l',
            dataset_name=Dataset.BA3Motifs,
            training_config=TrainingConfig(300, OptimType.ADAM, 0.01, batch_size=1),
            save_log_in_file=True,
            num_classes=3,
            save_visualization=True,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.BA3Motifs, 'explanation', 'gcn3l', 'gnnexplainer'),
            num_instances_to_visualize=20,
            explain_graph=True,
            mask_features=True,
            coff_edge_size=0.0,
            sparsity=0.0,
            node_color_setter=None,
            plt_legend=None,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            coff_node_feat_size=0.0)
        
        model = GCN_3l_BN(model_level='graph', dim_node=1, dim_hidden=20, num_classes=conf.num_classes)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/229_gcn3l_BA3Motifs/weights/61', map_location=conf.device))
        
        explainer = GNNExplainer(model, epochs=conf.training_config.num_epochs,
                 lr=conf.training_config.lr, explain_graph=conf.explain_graph, 
                 coff_edge_size=conf.coff_edge_size, coff_node_feat_size=conf.coff_node_feat_size)

        super().__init__(conf, model, explainer)