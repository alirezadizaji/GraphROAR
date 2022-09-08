from abc import ABC
import os
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def take_for_visualization(self, name) -> bool:
        """ return True if the given sample (determined by its name) is suitable for visualization """
        return True
