from dig.xgraph.method import SubgraphX
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.models import *
import torch

from GNN_Explainability.config.base_config import BaseConfig

from ....enums import Dataset
from ...main import MainEntrypoint
from ....utils.visualization import visualization


class Entrypoint(MainEntrypoint):
    def __init__(self):
        conf = BaseConfig(
            try_num=5,
            try_name='ba2motifs_subgraphx',
            dataset_name=Dataset.BA2Motifs,
        )
        conf.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        conf.num_epochs = 1
        conf.save_log_in_file = False
        conf.shuffle_training = True
        conf.batch_size = 1

        conf.base_model = GIN_3l(model_level='graph', dim_node=10, dim_hidden=300, num_classes=2)
        conf.base_model.load_state_dict(torch.load('./GNN_Explainability/checkpoints/ba2motifs_gin_3l.pt', map_location=conf.device))

        super(Entrypoint, self).__init__(conf)

    def run(self):
        # train explainer
        explainer = SubgraphX(self.conf.base_model, num_classes=2, device=self.conf.device, explain_graph=True,
                            reward_method='mc_l_shapley')
        max_nodes = 15
        for data in self.conf.train_loader:
            data = data[0].to(self.conf.device)
            y_pred = self.conf.base_model(data=data).argmax(-1).item()
            explain_result, related_preds = \
                explainer.explain(data.x, data.edge_index,
                                  max_nodes=max_nodes,
                                  label=y_pred)

            explain_result = explainer.read_from_MCTSInfo_list(explain_result)
            explanation = find_closest_node_result(explain_result, max_nodes=max_nodes)
            edge_mask = data.edge_index[0].cpu().apply_(lambda x: x in explanation.coalition).bool() & \
                        data.edge_index[1].cpu().apply_(lambda x: x in explanation.coalition).bool()
            edge_mask = edge_mask.float().numpy()
            print(edge_mask)