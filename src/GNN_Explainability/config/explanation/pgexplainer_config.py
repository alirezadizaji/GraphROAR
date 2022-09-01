from dataclasses import dataclass
from typing import Optional

from .explain_config import ExplainConfig

@dataclass
class PGExplainerConfig(ExplainConfig):
    coff_size: bool = 0.0
    coff_ent: float = 0.0
    t0: float = 5.0
    t1: float = 1.0
    sample_bias: float = 0.0
    topk_selection: int = 10
    explainer_load_dir: Optional[str] = None