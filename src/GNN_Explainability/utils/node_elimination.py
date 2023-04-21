import torch
from torch_geometric.data import Data

def isolated_node_elimination(g: Data, mask: torch.Tensor, return_remained_ix: bool = False) -> Data:
    r""" It removes the isolated nodes within the given graph and returns a new generated graph.

    Args:
        g (Data): _description_
        mask (torch.Tensor): _description_
        return_remained_ix (bool, optional): If `True`, then return the index of remaining nodes and edges within the original graph. Defaults to False.

    Returns:
        Data: _description_
    """
    masked_edges = g.edge_index[:, mask] 
    B = g.x.size(0)

    # If all edges are removed, then (just to enable training) instead of node eliminating, make all of them the same
    if masked_edges.numel() == 0:
        g.x = torch.zeros_like(g.x)
        remained = torch.arange(B).long()
        edges_remained = torch.Tensor([])
    else:
        node_mask = torch.zeros(B).bool()
        remained: torch.Tensor = torch.unique(masked_edges)
        node_mask[remained] = True
        g.x = g.x[node_mask]

        # let's nodes indices start from zero
        row, col = masked_edges
        node_indices, _ = torch.sort(torch.unique(masked_edges))
        mapping = torch.full((node_indices.max().item() + 1,), fill_value=torch.inf)
        mapping[node_indices] = torch.arange(node_indices.numel()).float()
        
        masked_edges = torch.stack([mapping[row], mapping[col]], dim=0).long()
        masked_edges = masked_edges.to(g.edge_index.device)

        # Get indices of edges in original graph which remains in the new graph
        org_row, org_col = g.edge_index
        edges_remained = torch.nonzero(
                torch.any(org_row[:, None] == row[None], 1) &
                torch.any(org_col[:, None] == col[None], 1)) \
                .flatten()

    g.edge_index = masked_edges

    if return_remained_ix:
        return g, remained, edges_remained
        
    return g