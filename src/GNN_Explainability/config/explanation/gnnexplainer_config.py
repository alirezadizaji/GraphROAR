from dataclasses import dataclass
from .explain_config import ExplainConfig

@dataclass
class GNNExplainerConfig(ExplainConfig):
    mask_features: bool
    coff_edge_size: float = 0.0
    coff_node_feat_size: float = 0.0