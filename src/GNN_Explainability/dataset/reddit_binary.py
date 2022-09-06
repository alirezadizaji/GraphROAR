from .base_dataset import GraphClsDataset
from ..enums import Dataset

class RedditDataset(GraphClsDataset):
    dataset_name: Dataset = Dataset.REDDIT_BINARY