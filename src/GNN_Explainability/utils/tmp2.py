from torch_geometric.data import DataLoader


from .edge_elimination import init_edge_eliminator
from .visualization import visualization
from ..enums.data_spec import DataSpec
from ..dataset.ba_2motifs import BA2MotifsDataset


if __name__ == "__main__":
    loader = DataLoader(BA2MotifsDataset(DataSpec.TRAIN), batch_size=1, shuffle=False)
    for x in loader:
        x = x[0]
        pos = visualization(x, f"{x.name[0]}_{x.y.item()}")
        x1 = init_edge_eliminator("../data/ba_2motifs/explanation/gnnexplainer", 0.5, False)(x)
        visualization(x1, x1.name[0], pos)
        x2 = init_edge_eliminator("../data/ba_2motifs/explanation/gnnexplainer", 0.9, False)(x)
        visualization(x2, x2.name[0], pos)