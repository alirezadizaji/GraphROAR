from .graph_cls_dataset import GraphClsDataset
from ..enums import Dataset


class MSRC9Dataset(GraphClsDataset):
    dataset_name: Dataset = Dataset.MSRC9
