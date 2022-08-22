from dataclasses import dataclass, field
from typing import List, Optional

from ..enums import dataset
from .base_config import BaseConfig

@dataclass
class ROARConfig(BaseConfig):
    edge_masks_load_dir: Optional[str]
    """ If given then edge masks are loaded from this directory, O.W. use random weighting """

    edge_mask_symmetric: bool = True
    """ If true then edge masks must be symmetric"""
    
    roar_ratios: List[float] = field(default_factory=list)