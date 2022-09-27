from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Dict, Any, List

from ...utils.visualization import NodeColor

from ..base_config import BaseConfig
if TYPE_CHECKING:
    from ...enums import Color

@dataclass
class ExplainConfig(BaseConfig):
    num_classes: int
    """ total number of classes exist within the dataset """

    save_visualization: bool 
    """ If true then save visualization as a png file """
    
    visualize_explainer_perf: bool
    """ If true then enable visualizing explainer performance with some random instances """

    edge_mask_save_dir: Optional[str]
    """ If given then save edge masks provided by explainer per instance """

    num_instances_to_visualize: int
    """ Number of random instances to be visualized for explanation """

    sparsity: float

    explain_graph: bool
    """ If true then explain graph, O.W. explain node """

    node_color_setter: Optional[NodeColor]
    """ node color setter used for visualization (NOTE: keys must be node's feature) """

    plt_legend: Optional[Dict['Color', str]]
    """ legend to display """