import os

from dig.xgraph.models import *
import torch

from ..gin3l.gnn_explainer import color_setter, legend
from .....config import SubgraphXConfig, TrainingConfig
from .....entrypoints.core.explanation.subgraphx import SubgraphXEntrypoint

from .....enums import *


class Entrypoint(SubgraphXEntrypoint):
    def __init__(self):
        conf = SubgraphXConfig(
            try_num=91,
            try_name='subgraphx',
            dataset_name=Dataset.MUTAG,
            training_config=TrainingConfig(100, OptimType.ADAM, batch_size=1),
            device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            num_classes=2,
            save_visualization=True,
            visualize_explainer_perf=True,
            num_instances_to_visualize=20,
            edge_mask_save_dir=os.path.join('..', 'data', 'ba_2motifs', 'explanation', 'gcn3l', 'subgraphx_70%'),
            sparsity=0.0,
            node_color_setter=color_setter,
            plt_legend=legend,
            explain_graph=True,
            reward_method='mc_shapley',
            get_max_nodes=(lambda data: int(data.x.size(0) * 0.7) + 1),
            n_rollout=10
        )

        model = GCN_3l_BN(model_level='graph', dim_node=7, dim_hidden=60, num_classes=2)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/13_gcn3l_MUTAG/weights/126', map_location=conf.device))

        # explainer will be initiated during explaining an instance
        explainer = None
        super().__init__(conf, model, explainer)