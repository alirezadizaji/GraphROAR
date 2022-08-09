from dataclasses import dataclass
from typing import Callable

from torch_geometric.data import Data

from .explain_config import ExplainConfig

@dataclass
class SubgraphXConfig(ExplainConfig):
    reward_method: str
    get_max_nodes: Callable[[Data], int]