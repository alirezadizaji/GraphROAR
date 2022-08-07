from ..enums import dataset
from .base_config import BaseConfig

class ROARConfig(BaseConfig):
    def __init__(self, try_num: int, try_name: str, dataset_name: dataset):
        super().__init__(try_num, try_name, dataset_name)

        self.edge_masks_root_dir: str = None
        """ the root directory where all edge masks from an explanation method are located """