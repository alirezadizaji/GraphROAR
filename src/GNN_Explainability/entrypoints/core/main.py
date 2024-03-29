from abc import ABC, abstractmethod
import os
from typing import TYPE_CHECKING, Tuple

from torch.optim import Adam, SGD
from torch_geometric.data import DataLoader

from ...dataset import *
from ...enums import *

if TYPE_CHECKING:
    from ...config.base_config import BaseConfig
    from dig.xgraph.models import GNNBasic
    from torch.optim import Optimizer

def get_dataset_cls(dataset: Dataset):
    if dataset == Dataset.BA2Motifs:
            cls = BA2MotifsDataset
    elif dataset == Dataset.MUTAG:
        cls = MUTAGDataset
    elif dataset == Dataset.REDDIT_BINARY:
        cls = RedditDataset
    elif dataset == Dataset.BA3Motifs:
        cls = BA3MotifsDataset
    elif dataset == Dataset.ENZYME:
        cls = EnzymeDataset
    elif dataset == Dataset.IMDB_BIN:
        cls = IMDBBinDataset
    else:
        raise NotImplementedError()
    
    return cls

class MainEntrypoint(ABC):
    def __init__(self, conf: 'BaseConfig', model: 'GNNBasic') -> None:
        self.conf = conf
        self.model = model
        self.model.to(self.conf.device)


        train, val, test = self.get_loaders()
        self.train_loader: 'DataLoader' = train
        self.val_loader: 'DataLoader' = val
        self.test_loader: 'DataLoader' = test

        self.optimizer: 'Optimizer' = self.get_optimizer()

        os.makedirs(self.conf.save_dir, exist_ok=True)
        
    def get_loaders(self) -> Tuple['DataLoader', 'DataLoader', 'DataLoader']:
        cls = get_dataset_cls(self.conf.dataset_name)
        train_set = cls(DataSpec.TRAIN)
        val_set = cls(DataSpec.VAL)
        test_set = cls(DataSpec.TEST)

        train = DataLoader(train_set, 
                    batch_size=self.conf.training_config.batch_size, 
                    shuffle=self.conf.training_config.shuffle_training)
        val = DataLoader(val_set, 
                    batch_size=self.conf.training_config.batch_size)
        test = DataLoader(test_set,
                    batch_size=self.conf.training_config.batch_size)
        return train, val, test

    def get_optimizer(self) -> 'Optimizer':
        if self.conf.training_config.optim_type == OptimType.ADAM:
            return Adam(self.model.parameters(), self.conf.training_config.lr)
        elif self.conf.training_config.optim_type == OptimType.SGD:
            return SGD(self.model.parameters(), self.conf.training_config.lr)
            
    @abstractmethod
    def run(self):
        pass