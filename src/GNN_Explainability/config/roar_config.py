from dataclasses import dataclass, field
from typing import List

from ..enums import dataset
from .base_config import BaseConfig

@dataclass
class ROARConfig(BaseConfig):
    edge_masks_load_dir: str 
    """ the root directory from which edge masks are loaded """

    roar_ratios: List[float] = field(default_factory=list)