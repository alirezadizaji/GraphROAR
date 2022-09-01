import os

from dig.xgraph.method import PGExplainer
from dig.xgraph.models import *
import torch


from .....config import PGExplainerConfig, TrainingConfig
from .....enums import *
from ....core import PGExplainerEntrypoint


class Entrypoint(PGExplainerEntrypoint):
    def __init__(self):
        conf = PGExplainerConfig(
            try_num=4,
            try_name='pgexplainer_gin3l',
            dataset_name=Dataset.BA2Motifs,
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            training_config=TrainingConfig(30, OptimType.ADAM, lr=3e-3, batch_size=1),
            num_classes=2,
            save_visualization=True,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', 'ba_2motifs', 'explanation', 'pgexplainer'),
            num_instances_to_visualize=20,
            sparsity=0.0,
            explain_graph=True,
            node_color_setter=None,
            plt_legend=None,
            explainer_load_dir=os.path.join('..', 'results', '4_pgexplainer_gin3l_BA2Motifs', 'weights', 'model.pt')
        )

        model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        model.load_state_dict(torch.load('../checkpoints/ba2motifs_gin_3l.pt', map_location=conf.device))
 
        explainer = PGExplainer(model, in_channels=600, 
                device=conf.device, explain_graph=True,
                epochs=conf.training_config.num_epochs, 
                lr=conf.training_config.lr, coff_size=conf.coff_size,
                coff_ent=conf.coff_ent, t0=conf.t0, t1=conf.t1,
                sample_bias=conf.sample_bias)
        
        super(Entrypoint, self).__init__(conf, model, explainer)