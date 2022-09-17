from .graph_cls_dataset import GraphClsDataset
from ..enums import Dataset


class IMDBBinDataset(GraphClsDataset):
    dataset_name: Dataset = Dataset.IMDB_BIN
