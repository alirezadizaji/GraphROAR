import os

from dig.xgraph.method import PGExplainer
from dig.xgraph.models import *
import torch

from ..gin3l.gnn_explainer import color_setter, legend
from .....config import PGExplainerConfig, TrainingConfig
from .....enums import *
from ....core import PGExplainerEntrypoint
from .....utils.symmetric_edge_mask import symmetric_edges


class Entrypoint(PGExplainerEntrypoint):
    def __init__(self):
        conf = PGExplainerConfig(
            try_num=473,
            try_name='pgexplainer_gcn3l',
            dataset_name=Dataset.MSRC9,
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
            save_log_in_file=True,
            training_config=TrainingConfig(30, OptimType.ADAM, lr=3e-3, batch_size=1),
            num_classes=8,
            save_visualization=True,
            visualize_explainer_perf=True,
            edge_mask_save_dir=os.path.join('..', 'data', Dataset.MSRC9, 'explanation', 'gcn3l', 'pgexplainer'),
            num_instances_to_visualize=50,
            sparsity=0.0,
            explain_graph=True,
            node_color_setter=color_setter,
            plt_legend=legend,
        )

        model = GCN_3l_BN(model_level='graph', dim_node=10, dim_hidden=20, num_classes=8)
        model.to(conf.device)
        model.load_state_dict(torch.load('../results/462_gcn3l_MSRC_9/weights/41', map_location=conf.device))
   
        explainer = PGExplainer(model, in_channels=40, 
                device=conf.device, explain_graph=conf.explain_graph,
                epochs=conf.training_config.num_epochs, 
                lr=conf.training_config.lr, coff_size=conf.coff_size,
                coff_ent=conf.coff_ent, t0=conf.t0, t1=conf.t1,
                sample_bias=conf.sample_bias)
        
        super(Entrypoint, self).__init__(conf, model, explainer)

    def _select_explainable_edges(self, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        edge_mask = symmetric_edges(edge_index, edge_mask)
        k = int(edge_mask.numel() * 0.3)
        edge_index = edge_index[:, edge_mask.topk(k)[1]]

        return edge_index