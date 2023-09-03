import os
import numpy as np
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def dist_send_sparse(x, dst, tag):
    """
    x : torch.tensor
        To fast computation, x shold be on GPU.

    dst : int

    tag : int
    """
    
    n_param_send = 0
    
    sparse_x = x.view(-1).to_sparse().coalesce()
    x_indices = sparse_x.indices().to(torch.int32).contiguous()
    x_values = sparse_x.values()

    # exchange the number of non-zero elements.
    n_elem = torch.torch.numel(x_indices)
    dist.send(torch.tensor(n_elem, dtype=torch.int32), dst=dst, tag=tag)
    n_param_send += 1
    
    # exchange indices and values.
    dist.send(x_indices.to("cpu"), dst=dst, tag=tag)
    dist.send(x_values.to("cpu"), dst=dst, tag=tag)
    n_param_send += torch.numel(x_indices) + torch.numel(x_values)

    return n_param_send


def dist_recv_sparse(x_size, src, tag, device):
    """
    Parameter
    ----------
    x_size : torch.Size
        the size of dual_z or dual_y

    src : int
        the id of source node.

    tag : int

    device : torch.Device
    
    Return 
    ----------
    torch.Tensor (shape (1, #non-zero))
        indecies of non-zero elements. (i.e., COO format).
        this tensor is on 'device'.

    torch.Tensor
        the received 'comp(dual_y)'.
        this tensor is on 'device'.
    """

    # exchange the number of non-zero elements.
    n_elem = torch.zeros(1, dtype=torch.int32)
    dist.recv(n_elem, src=src, tag=tag)
    n_elem = n_elem.item()

    # exchange indices and values.
    x_indices = torch.zeros((1, n_elem), dtype=torch.int32)
    x_values = torch.zeros(n_elem)
    dist.recv(x_indices, src=src, tag=tag)
    dist.recv(x_values, src=src, tag=tag)

    x_indices, x_values = x_indices.to(device), x_values.to(device)
    x_indices = x_indices.to(torch.int64)
    
    return x_indices, torch.sparse.FloatTensor(x_indices, x_values, [np.prod(x_size)]).to_dense().view(x_size)


def recv_and_compute_diff(dual_z, src, tag):
    """
    Receive dual_y from node 'src', then return 'comp(dual_y - dual_z)'.
    """
    
    mask_indices, adj_dual_y = dist_recv_sparse(dual_z.size(), src, tag, dual_z.device)

    masked_dual_z = torch.sparse.FloatTensor(mask_indices.to(torch.int64), dual_z.view(-1)[mask_indices[0]], [torch.numel(dual_z)]).to_dense().view(dual_z.size())

    return adj_dual_y - masked_dual_z

