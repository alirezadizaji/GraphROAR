import os

from dig.xgraph.method import SubgraphX
from dig.xgraph.models import *
import torch


from ....config import SubgraphXConfig, TrainingConfig
from ....entrypoints.core.explanation.subgraphx import SubgraphXEntrypoint

from ....enums import *


class Entrypoint(SubgraphXEntrypoint):
    def __init__(self):
        conf = SubgraphXConfig(
            try_num=7,
            try_name='ba2motifs_subgraphx',
            dataset_name=Dataset.BA2Motifs,
            training_config=TrainingConfig(100, OptimType.ADAM, batch_size=1),
            device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=False,
            num_classes=2,
            save_visualization=True,
            visualize_explainer_perf=True,
            num_instances_to_visualize=20,
            edge_mask_save_dir=os.path.join('..', 'data', 'ba_2motifs', 'explanation', 'subgraphx_10%'),
            sparsity=0.0,
            explain_graph=True,
            reward_method='mc_l_shapley',
            get_max_nodes=(lambda data: int(data.edge_index.size(1) * 0.1) + 1)

        )

        model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        model.load_state_dict(torch.load('../GNN_Explainability/checkpoints/ba2motifs_gin_3l.pt', map_location=conf.device))

        explainer = SubgraphX(model, num_classes=conf.num_classes, device=conf.device, 
                        explain_graph=conf.explain_graph, reward_method=conf.reward_method)
        
        super().__init__(conf, model, explainer)