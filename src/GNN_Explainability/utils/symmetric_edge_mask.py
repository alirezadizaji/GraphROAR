import torch

def symmetric_edges(edge_index: 'torch.Tensor', edge_mask: 'torch.Tensor') -> 'torch.Tensor':
    """ makes the given edge_mask symmetric. this is suitable for indirect graphs """
    
    edge_mask = edge_mask.to(edge_index.device)

    num_nodes = edge_index.unique().numel()
    edge_mask_asym = torch.sparse_coo_tensor(edge_index, 
            edge_mask, (num_nodes, num_nodes)).to_dense()
    edge_mask_sym = (edge_mask_asym + edge_mask_asym.T) / 2
    edge_mask = edge_mask_sym[edge_index[0], edge_index[1]]

    return edge_mask