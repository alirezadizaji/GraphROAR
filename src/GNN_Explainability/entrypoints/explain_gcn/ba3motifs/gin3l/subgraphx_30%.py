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
            try_num=243,
            try_name='subgraphx',
            dataset_name=Dataset.BA3Motifs,
            training_config=TrainingConfig(100, OptimType.ADAM, batch_size=1),
            device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            num_classes=3,
            save_visualization=True,
            visualize_explainer_perf=True,
            num_instances_to_visualize=20,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.BA3Motifs, 'explanation', 'gin3l', 'subgraphx_30%'),
            sparsity=0.0,
            node_color_setter=None,
            plt_legend=None,
            explain_graph=True,
            reward_method='mc_shapley',
            get_max_nodes=(lambda data: int(data.edge_index.size(1)/2 * 0.3) + 1),
            n_rollout=20,
        )

        model = GIN_3l(model_level='graph', dim_node=1, dim_hidden=20, num_classes=3)
        model.load_state_dict(torch.load('../results/230_gin3l_BA3Motifs/weights/89', map_location=conf.device))

        # explainer will be initiated during explaining an instance
        explainer = None
        super().__init__(conf, model, explainer)