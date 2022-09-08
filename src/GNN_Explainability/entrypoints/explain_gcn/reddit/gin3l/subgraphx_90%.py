import os

from dig.xgraph.models import *
import torch

from .....config import SubgraphXConfig, TrainingConfig
from .....entrypoints.core.explanation.subgraphx import SubgraphXEntrypoint

from .....enums import *


class Entrypoint(SubgraphXEntrypoint):
    def __init__(self):
        conf = SubgraphXConfig(
            try_num=196,
            try_name='subgraphx',
            dataset_name=Dataset.REDDIT_BINARY,
            training_config=TrainingConfig(100, OptimType.ADAM, batch_size=1),
            device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            num_classes=2,
            save_visualization=True,
            visualize_explainer_perf=True,
            num_instances_to_visualize=20,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.REDDIT_BINARY, 'explanation', 'gin3l', 'subgraphx_90%'),
            sparsity=0.0,
            explain_graph=True,
            reward_method='mc_shapley',
            get_max_nodes=(lambda data: int(data.x.size(0) * 0.9) + 1),
            n_rollout=10,
            plt_legend=None,
            node_color_setter=None,
        )

        model = GIN_3l(model_level='graph', dim_node=1, dim_hidden=60, num_classes=conf.num_classes)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/180_gin3l_REDDIT-BINARY/weights/208', map_location=conf.device))

        # explainer will be initiated during explaining an instance
        explainer = None
        super().__init__(conf, model, explainer)