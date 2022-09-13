from .graph_cls_dataset import GraphClsDataset
from ..enums import Dataset


class BA3MotifsDataset(GraphClsDataset):
    dataset_name: Dataset = Dataset.BA3Motifs
