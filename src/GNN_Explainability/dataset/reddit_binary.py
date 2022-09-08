from typing import List

import numpy as np

from .graph_cls_dataset import GraphClsDataset
from ..enums import Dataset

class RedditDataset(GraphClsDataset):
    dataset_name: Dataset = Dataset.REDDIT_BINARY
    names_to_be_visualized: List[str] = None

    def take_for_visualization(self, name) -> bool:
        if RedditDataset.names_to_be_visualized is None:   
            graph_num_nodes = np.array([g.x.shape[0] for g in self.graphs])
            indices = np.argsort(graph_num_nodes)[:100]
            
            ys = np.array([self.graphs[i].y.item() for i in indices])
            idx1 = np.nonzero(ys == 0)[0][:10]
            idx2 = np.nonzero(ys == 1)[0][:10]
            indices = np.concatenate([indices[idx1], indices[idx2]])
            
            RedditDataset.names_to_be_visualized = [self.graphs[i].name for i in indices]
        
        return name in RedditDataset.names_to_be_visualized