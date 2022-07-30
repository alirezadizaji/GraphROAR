from typing import Optional, TYPE_CHECKING

from torch_geometric.data import DataLoader

from ..dataset import *
from ..enums import *

if TYPE_CHECKING:
    from dig.xgraph.models import GNNBasic
    from torch.optim import Optimizer


class BaseConfig:
    def __init__(self, try_num: int, try_name: str, dataset_name: Dataset):
        self.try_num: int = try_num
        """ number of experiment """

        self.try_name: str = try_name
        """ name of experiment """

        self.dataset_name: Dataset = dataset_name
        """ dataset name, to fill dataloader with """

        self.device = None
        """ device to run """

        self.save_log_in_file: bool = True
        """ if True then save logs in a file, O.W. in terminal """

        self.base_model: 'GNNBasic' = None
        """ base model to check its explantion"""

        self.save_dir: Optional[str] = None
        r""" if given, then save logs into this directory, O.W. use `try_num` and `try_name` for this purpose """        
        
        self.optimizer: 'Optimizer' = None
        self.num_epochs: int = None

        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None

        self.set_loaders()


    def set_loaders(self):
        if self.dataset_name == Dataset.BA2Motifs:
            self.train_loader = DataLoader(BA2MotifsDataset(DataSpec.TRAIN), shuffle=False)
            self.val_loader = DataLoader(BA2MotifsDataset(DataSpec.VAL))
            self.test_loader = DataLoader(BA2MotifsDataset(DataSpec.TEST))