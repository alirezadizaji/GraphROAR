from typing import List

import numpy as np

from .graph_cls_dataset import GraphClsDataset
from ..enums import Dataset

class EnzymeDataset(GraphClsDataset):
    dataset_name: Dataset = Dataset.ENZYME