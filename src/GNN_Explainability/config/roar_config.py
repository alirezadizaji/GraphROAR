from dataclasses import dataclass, field
from typing import List, Optional

from ..enums import dataset
from .base_config import BaseConfig

@dataclass
class ROARConfig(BaseConfig):
    edge_masks_load_dir: str
    """ the location where the edge masks are going to be loaded from """

    edge_mask_symmetric: bool = True
    """ If true then edge masks must be symmetric """
    
    edge_mask_random_weighting: bool = False
    r""" if True then create random edge mask and save it in `edge_masks_load_dir` """
    
    roar_ratios: List[float] = field(default_factory=list)