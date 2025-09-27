import torch_geometric
import torch
import numpy as np
from scipy import sparse as sp


def random_walk(A, n_iter):
    # Geometric diffusion features with Random Walk
    Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
    RW = A * Dinv
    M = RW
    M_power = M
    # Iterate
    PE = [torch.diagonal(M)]
    for _ in range(n_iter - 1):
        M_power = torch.matmul(M_power, M)
        PE.append(torch.diagonal(M_power))
    PE = torch.stack(PE, dim=-1)
    return PE


def RWSE(edge_index, pos_enc_dim, num_nodes):
    """
        Initializing positional encoding with RWSE
    """
    if edge_index.size(-1) == 0:
        PE = torch.zeros(num_nodes, pos_enc_dim)
    else:
        A = torch_geometric.utils.to_dense_adj(
            edge_index, max_num_nodes=num_nodes)[0]
        PE = random_walk(A, pos_enc_dim)
    return PE


def LapPE(edge_index, pos_enc_dim, num_nodes):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    degree = torch_geometric.utils.degree(edge_index[0], num_nodes)
    A = torch_geometric.utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes)
    N = sp.diags(np.array(degree.clip(1) ** -0.5, dtype=float))
    L = sp.eye(num_nodes) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    PE = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    if PE.size(1) < pos_enc_dim:
        zeros = torch.zeros(num_nodes, pos_enc_dim)
        zeros[:, :PE.size(1)] = PE
        PE = zeros
    return PE


def pad_coarsen_adj(coarsen_adj, max_patches):
    """
    Pad coarsen_adj to (max_patches, max_patches) with zeros.
    Args:
        coarsen_adj: (patch_num, patch_num)
        max_patches: int
    Returns:
        padded_coarsen_adj: (max_patches, max_patches)
        key_padding_mask: (max_patches), where True indicates padding
    """
    patch_num = coarsen_adj.size(0)
    device = coarsen_adj.device

    # Pad coarsen_adj
    padded_coarsen_adj = torch.zeros(max_patches, max_patches, device=device)
    padded_coarsen_adj[:patch_num, :patch_num] = coarsen_adj

    # Generate key_padding_mask (True for padding positions)
    # key_padding_mask = torch.arange(max_patches, device=device) >= patch_num

    # return padded_coarsen_adj, key_padding_mask
    return padded_coarsen_adj

