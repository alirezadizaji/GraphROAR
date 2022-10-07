from dataclasses import dataclass, field
from typing import List, Optional

from ..enums import dataset
from .base_config import BaseConfig

@dataclass
class ROARConfig(BaseConfig):
    edge_masks_load_dir: str
    """ the location where the probability edge weightings are going to be loaded from """

    edge_mask_symmetric: bool = True
    """ If true then edge weightings must be symmetric """
    
    edge_mask_random_weighting: bool = False
    r""" if True then create random probability edge weightings and save it in `edge_masks_load_dir` """
    
    eliminate_top_most_edges: bool = True
    """ If True then eliminate top most edges (roar) otherwise keep them and eliminate the rest (kar) """
    
    skip_during_evaluation: bool = False
    """ If True then skip ROAR during evaluation (validation and test phases), O.W. apply it on them too """
    
    eliminate_nodes_too: bool = False
    """ If True then eliminate nodes whose all connected edges have been removed. """
    
    prob_to_replace_instead_of_remove: Optional[float] = None
    """ A probability threshold to randomly replace some edges instead of removing them. If not given, then do not apply replacement."""
    
    roar_ratios: List[float] = field(default_factory=list)
    """ ratios (from 0.0 to 1.0) to perform retraining steps """