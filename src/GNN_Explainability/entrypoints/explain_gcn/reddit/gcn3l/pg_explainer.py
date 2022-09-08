import os

from dig.xgraph.method import PGExplainer
from dig.xgraph.models import *
import torch

from .....utils.symmetric_edge_mask import symmetric_edges
from .....config import PGExplainerConfig, TrainingConfig
from .....enums import *
from ....core import PGExplainerEntrypoint


class Entrypoint(PGExplainerEntrypoint):
    def __init__(self):
        conf = PGExplainerConfig(
            try_num=183,
            try_name='pgexplainer',
            dataset_name=Dataset.REDDIT_BINARY,
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            training_config=TrainingConfig(15, OptimType.ADAM, lr=3e-3, batch_size=1),
            num_classes=2,
            save_visualization=True,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.REDDIT_BINARY, 'explanation', 'gcn3l', 'pgexplainer'),
            num_instances_to_visualize=20,
            sparsity=0.0,
            explain_graph=True,
            plt_legend=None,
            node_color_setter=None,
            explainer_load_dir='../results/183_pgexplainer_REDDIT-BINARY/weights/model.pt')        

        model = GCN_3l_BN(model_level='graph', dim_node=1, dim_hidden=60, num_classes=2)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/179_gcn3l_REDDIT-BINARY/weights/450', map_location=conf.device))
 
        explainer = PGExplainer(model, in_channels=120, 
                device=conf.device, explain_graph=conf.explain_graph,
                epochs=conf.training_config.num_epochs, 
                lr=conf.training_config.lr, coff_size=conf.coff_size,
                coff_ent=conf.coff_ent, t0=conf.t0, t1=conf.t1,
                sample_bias=conf.sample_bias)
        
        super(Entrypoint, self).__init__(conf, model, explainer)
    
    def _select_explainable_edges(self, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        edge_mask = symmetric_edges(edge_index, edge_mask)
        k = int(edge_mask.numel() * 0.5)
        edge_index = edge_index[:, edge_mask.topk(k)[1]]

        return edge_index