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
            try_num=235,
            try_name='subgraphx_gcn3l',
            dataset_name=Dataset.BA3Motifs,
            training_config=TrainingConfig(100, OptimType.ADAM, batch_size=1),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            num_classes=3,
            save_visualization=True,
            visualize_explainer_perf=True,
            num_instances_to_visualize=20,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.BA3Motifs, 'explanation', 'gcn3l', 'subgraphx_30%'),
            sparsity=0.0,
            node_color_setter=None,
            plt_legend=None,
            explain_graph=True,
            reward_method='mc_shapley',
            get_max_nodes=(lambda data: int(data.x.size(0) * 0.3) + 1),
            n_rollout=10,
        )

        model = GCN_3l_BN(model_level='graph', dim_node=1, dim_hidden=20, num_classes=3)
        model.load_state_dict(torch.load('../results/229_gcn3l_BA3Motifs/weights/61', map_location=conf.device))

        # explainer will be initiated during explaining an instance
        explainer = None
        super().__init__(conf, model, explainer)