from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..enums import Dataset, OptimType


@dataclass
class BaseConfig:
    try_num: int
    """ number of experiment """

    try_name: str
    """ name of experiment """

    dataset_name: 'Dataset'
    """ dataset name, to fill dataloader with """

    training_config: 'TrainingConfig' 
    """ config of your training including optimizer, number of epochs and data shuffling """

    device: str
    """ device to run """

    save_log_in_file: bool
    """ if True then save logs in a file, O.W. in terminal """


    @property
    def save_dir(self):
        folder_name = "_".join([self.try_name, self.try_num, self.dataset_name])
        root = "../results"
        return os.path.join(root, folder_name)

@dataclass
class TrainingConfig:
    num_epochs: int
    optim_type: 'OptimType'
    lr: float = 1e-3
    shuffle_training: bool = True
    batch_size: int = 32