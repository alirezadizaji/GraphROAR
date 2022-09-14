import os

from dig.xgraph.method import SubgraphX
from dig.xgraph.models import *
import torch


from .....config import SubgraphXConfig, TrainingConfig
from .....entrypoints.core.explanation.subgraphx import SubgraphXEntrypoint

from .....enums import *


class Entrypoint(SubgraphXEntrypoint):
    def __init__(self):
        conf = SubgraphXConfig(
            try_num=296,
            try_name='subgraphx',
            dataset_name=Dataset.ENZYME,
            training_config=TrainingConfig(100, OptimType.ADAM, batch_size=1),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            num_classes=6,
            save_visualization=True,
            visualize_explainer_perf=True,
            num_instances_to_visualize=20,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.ENZYME, 'explanation', 'gin3l', 'subgraphx_10%'),
            sparsity=0.0,
            node_color_setter=None,
            plt_legend=None,
            explain_graph=True,
            reward_method='mc_shapley',
            get_max_nodes=(lambda data: int(data.x.size(0) * 0.1) + 1),
            n_rollout=10,
        )

        model = GIN_3l(model_level='graph', dim_node=18, dim_hidden=80, num_classes=conf.num_classes)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/284_gin3l_ENZYMES/weights/404', map_location=conf.device))

        # explainer will be initiated during explaining an instance
        explainer = None
        super().__init__(conf, model, explainer)