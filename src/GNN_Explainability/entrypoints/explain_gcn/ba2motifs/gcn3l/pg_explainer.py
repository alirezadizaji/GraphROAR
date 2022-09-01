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
            try_num=39,
            try_name='pgexplainer_gcn3l',
            dataset_name=Dataset.BA2Motifs,
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=False,
            training_config=TrainingConfig(30, OptimType.ADAM, lr=3e-3, batch_size=1),
            num_classes=2,
            save_visualization=False,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', 'ba_2motifs', 'explanation', 'gcn3l', 'pgexplainer'),
            num_instances_to_visualize=20,
            sparsity=0.0,
            explain_graph=True,
            node_color_setter=None,
            plt_legend=None,
            explainer_load_dir=os.path.join('..', 'results', '39_pgexplainer_gcn3l_BA2Motifs', 'weights', 'model.pt')
        )

        model = GCN_3l_BN(model_level='graph', dim_node=10, dim_hidden=20, num_classes=2)
        model.load_state_dict(torch.load('../results/1_gcn3l_BA2Motifs/weights/16', map_location=conf.device))
        
        explainer = PGExplainer(model, in_channels=40, 
                device=conf.device, explain_graph=conf.explain_graph,
                epochs=conf.training_config.num_epochs, 
                lr=conf.training_config.lr, coff_size=conf.coff_size,
                coff_ent=conf.coff_ent, t0=conf.t0, t1=conf.t1,
                sample_bias=conf.sample_bias)
        
        super(Entrypoint, self).__init__(conf, model, explainer)