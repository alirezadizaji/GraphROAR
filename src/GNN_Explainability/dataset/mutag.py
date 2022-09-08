from .graph_cls_dataset import GraphClsDataset
from ..enums import Dataset


class MUTAGDataset(GraphClsDataset):
    dataset_name: Dataset = Dataset.MUTAG
