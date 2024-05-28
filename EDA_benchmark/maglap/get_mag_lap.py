from typing import Any, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
    to_dense_adj
)

from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

@functional_transform('add_mag_laplacian_eigenvector_pe')
class AddMagLaplacianEigenvectorPE(BaseTransform):
    r"""Adds the Magnetic Laplacian eigenvector positional encoding. The eigenvectors are
    complex number, so choosing k of them means there will be 2*k channels (k real parts and k imaginary parts)
    in total.

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(
            self,
            k: int,
            q: float = 0.1,
            dynamic_q: bool = False,
            multiple_q: int = 1,
            attr_name: Optional[str] = 'laplacian_eigenvector_pe',
            **kwargs,
    ):
        self.k = k
        self.q = q
        self.dynamic_q = dynamic_q
        self.multiple_q = multiple_q
        self.attr_name = attr_name
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        from scipy.sparse.linalg import eigs, eigsh
        eig_fn = eigsh # always use hermitian version

        num_nodes = data.num_nodes
        edge_index, edge_weight_list = get_mag_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
            q = self.q,
            dynamic_q=self.dynamic_q,
            multiple_q=self.multiple_q
        )

        pe_list = []
        eigvals_list = []
        for edge_weight in edge_weight_list:
            L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

            #try:
            #    eig_vals, eig_vecs = eig_fn(
            #        L,
            #        k=self.k,
            #        which='SA',
            #        return_eigenvectors=True,
            #        **self.kwargs,
            #    )
            #    sort = eig_vals.argsort()
            #    eig_vals = eig_vals[sort]
            #    eig_vecs = eig_vecs[:, sort]
            #except:
                #from scipy.linalg import eigh
                #eig_vals, eig_vecs = eigh(L.toarray())
                #sort = eig_vals.argsort()[:self.k]
                #eig_vals = eig_vals[sort]
                #eig_vecs = eig_vecs[:, sort]
                #eig_vals = eig_vals[:self.k]
                #eig_vecs = eig_vecs[:, :self.k]

            #if np.isnan(eig_vecs).any() or np.isnan(eig_vals).any():
            eig_vals, eig_vecs = np.linalg.eigh(L.toarray())
            sort = eig_vals.argsort()[:self.k]
            eig_vals = eig_vals[sort]
            eig_vecs = eig_vecs[:, sort]

            # padding zeros if num of nodes less than desired pe dimension
            if len(eig_vals) < self.k:
                eig_vals = np.pad(eig_vals, (0, self.k - len(eig_vals)))
                eig_vecs = np.pad(eig_vecs, ((0, 0),(0, self.k - eig_vecs.shape[-1])))

            #pe = np.concatenate( (np.expand_dims(np.real(eig_vecs[:, eig_vals.argsort()]), -1),
            #                           np.expand_dims(np.imag(eig_vecs[:, eig_vals.argsort()]), -1)), axis=-1)
            #pe = np.concatenate( (np.expand_dims(np.real(eig_vecs), -1),
            #                           np.expand_dims(np.imag(eig_vecs), -1)), axis=-1)
            # pe = torch.from_numpy(pe) # [N, pe_dim, 2]
            #sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
            #sign = torch.unsqueeze(torch.unsqueeze(sign, dim=-1), dim=0)
            #pe = sign * pe

            #pe = pe.flatten(1, 2) # [N, pe_dim * 2]



            pe = torch.from_numpy(np.expand_dims(eig_vecs, 1))
            eig_vals = np.expand_dims(np.expand_dims(eig_vals, 0), 0)
            pe_list.append(pe)
            eigvals_list.append(torch.from_numpy(eig_vals))
        #pe = torch.cat(pe_list, dim=-1)
        #eig_vals = torch.cat(eigvals_list, dim=-1)
        pe = torch.cat(pe_list, dim=1)
        eig_vals = torch.cat(eigvals_list, dim=1)
        data = add_node_attr(data, pe, attr_name=self.attr_name)
        #data = add_node_attr(data, eig_vals.reshape(1, -1), attr_name='Lambda')
        data = add_node_attr(data, eig_vals, attr_name='Lambda')
        return data




def get_mag_laplacian(
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        normalization: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        num_nodes: Optional[int] = None,
        q = 0.1,
        dynamic_q: Optional[bool] = False,
        multiple_q: Optional[int] = 1,
) -> Tuple[Tensor, OptTensor]:
    r""" Computes the graph Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        dtype (torch.dtype, optional): The desired data type of returned tensor
            in case :obj:`edge_weight=None`. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2],
        ...                            [1, 0, 2, 1]])
        >>> edge_weight = torch.tensor([1., 2., 2., 4.])

        >>> # No normalization
        >>> lap = get_laplacian(edge_index, edge_weight)

        >>> # Symmetric normalization
        >>> lap_sym = get_laplacian(edge_index, edge_weight,
                                    normalization='sym')

        >>> # Random-walk normalization
        >>> lap_rw = get_laplacian(edge_index, edge_weight, normalization='rw')
    """

    if normalization is not None:
        assert normalization in ['sym', 'rw']  # 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum') + \
          scatter(edge_weight, col, 0, dim_size=num_nodes, reduce='sum') # symmetric total degree

    # apply a complex adjacency matrix
    if dynamic_q: # normalize q using longest path distance
        adj_matrix = to_dense_adj(edge_index)
        l = (adj_matrix != adj_matrix.transpose(1, 2)).sum() / 2
        q /= max(min(l.item(), num_nodes), 1)

    q0 = q
    edge_weight0 = edge_weight
    edge_index0 = edge_index
    edge_weight_list = []
    for i in range(1, multiple_q+1):
        if multiple_q == 1:
            q = q0
        else:
            q = q0 * i
        #if q == 0:
        #    edge_weight = edge_weight0
        #else:
        edge_weight = torch.tensor(np.exp(1j * 2 * np.pi * q)) * edge_weight0

        if normalization is None:
            # L = D - A.
            edge_index = torch.cat([edge_index0, edge_index0[[1, 0]]], dim=-1) # symmetrization
            edge_weight = torch.cat([edge_weight, torch.conj(edge_weight)], dim=0) # complex conjugate the reverse edges
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            edge_weight = torch.cat([-edge_weight, deg], dim=0)
        elif normalization == 'sym':
            # Compute A_norm = -D^{-1/2} A D^{-1/2}.
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


            edge_index = torch.cat([edge_index0, edge_index0[[1, 0]]], dim=-1) # symmetrization
            edge_weight = torch.cat([edge_weight, torch.conj(edge_weight)], dim=0) # complex conjugate the reverse edges

            # L = I - A_norm.

            edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                             fill_value=1., num_nodes=num_nodes)
            assert tmp is not None
            edge_weight = tmp
        else:
            # Compute A_norm = -D^{-1} A.
            deg_inv = 1.0 / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            edge_weight = deg_inv[row] * edge_weight

            # L = I - A_norm.
            edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                             fill_value=1., num_nodes=num_nodes)
            assert tmp is not None
            edge_weight = tmp
        edge_weight_list.append(edge_weight)

    return edge_index, edge_weight_list


class AddLaplacianEigenvectorPE(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(
            self,
            k: int,
            attr_name: Optional[str] = 'laplacian_eigenvector_pe',
            is_undirected: bool = False,
            **kwargs,
    ):
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        from scipy.sparse.linalg import eigs, eigsh
        eig_fn = eigsh

        num_nodes = data.num_nodes
        # symmetrization: always make graph undirected
        edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=-1)
        edge_index, edge_weight = get_laplacian(
            edge_index,
            edge_weight=None,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        '''try:
            eig_vals, eig_vecs = eig_fn(
                L,
                k=self.k + 1,
                which='SR' if not self.is_undirected else 'SA',
                return_eigenvectors=True,
                **self.kwargs,
            )
            #eig_vals, eig_vecs = np.linalg.eig(L)
            sort = eig_vals.argsort()
            eig_vals = eig_vals[sort]
            eig_vecs = eig_vecs[:, sort]
        except:'''
        # either L is not sparse or k >= num_nodes
        from scipy.linalg import eigh
        #eig_vals, eig_vecs = eigh(L.toarray())
        eig_vals, eig_vecs = np.linalg.eigh(L.toarray())
        sort = eig_vals.argsort()[:self.k+1]
        eig_vals = eig_vals[sort]
        eig_vecs = eig_vecs[:, sort]
        #eig_vals = eig_vals[:self.k+1]
        #eig_vecs = eig_vecs[:, :self.k+1]

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])


        if len(eig_vals) < self.k+1:
            eig_vals = np.pad(eig_vals, (0, self.k+1 - len(eig_vals)))
            eig_vecs = np.pad(eig_vecs, ((0, 0),(0, self.k+1 - eig_vecs.shape[-1])))

        pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe *= sign

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        data = add_node_attr(data, torch.from_numpy(eig_vals[1:self.k+1].reshape(1, -1)), attr_name='Lambda')
        return data


class AddSingularValuePE(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(
            self,
            k: int,
            attr_name: Optional[str] = 'laplacian_eigenvector_pe',
            is_undirected: bool = False,
            **kwargs,
    ):
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        from scipy.sparse.linalg import svds
        eig_fn = svds

        num_nodes = data.num_nodes
        # symmetrization: always make graph undirected
        edge_index = data.edge_index
        edge_index, edge_weight = get_laplacian(
            edge_index,
            edge_weight=None,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        try:
            U, eig_vals, V = eig_fn(
                L,
                k=self.k,
                which='SM',
                **self.kwargs,
            )
        except:
            from scipy.linalg import svd
            U, eig_vals, V = svd(L.toarray())
            # note that by default svd returns singular value from largest to smallest, so reverse it
            U, eig_vals, V = U[:, ::-1], eig_vals[::-1], V[:, ::-1]
            eig_vals = eig_vals[:self.k]
            U = U[:, :self.k]
            V = V[:, :self.k]

        #eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])


        if len(eig_vals) < self.k:
            eig_vals = np.pad(eig_vals, (0, self.k - len(eig_vals)))
            U = np.pad(U, ((0, 0),(0, self.k - U.shape[-1])))
            V = np.pad(U, ((0, 0),(0, self.k - V.shape[-1])))

        U = torch.from_numpy(U[:, :self.k])
        V = torch.from_numpy(V[:, :self.k])
        pe = torch.cat([U, V], dim=-1)
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe *= sign

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        data = add_node_attr(data, torch.from_numpy(eig_vals[0:self.k].reshape(1, -1)), attr_name='Lambda')
        return data


def get_walk_profile_v2(edge_index, M, output='spd'):
    # input: (directed) edge index
    # output: walk profile matrix up to M steps

    # step 1: construct walk sequence (A_q, A^2_q, A^3_q, ..., A^M_q)
    edge_index = remove_self_loops(edge_index)[0]
    A = to_dense_adj(edge_index)[0]
    A_q = torch.tile(torch.unsqueeze(A, 0), (M+1, 1, 1))
    phase_factor = torch.exp(1j * 2 * torch.pi * torch.tensor([j / 2 / (M+1) for j in range(0, M+1)])).to(edge_index.device)
    A_q = A_q * phase_factor.unsqueeze(-1).unsqueeze(-1) # [M+1, N, N]
    A_q = A_q + torch.conj(torch.transpose(A_q, 1, 2))
    A_q_m_power = torch.tile(torch.eye(A_q.size(-1), A_q.size(-1)).unsqueeze(0), (M+1, 1, 1)).to(edge_index.device) # [M+1, N, N]
    A_q_m_power = torch.complex(A_q_m_power, torch.zeros_like(A_q_m_power))
    A_q_all_power = []
    for m in range(1, M+1):
        A_q_m_power = torch.bmm(A_q_m_power, A_q) * (phase_factor.unsqueeze(-1).unsqueeze(-1)) # [M+1, N, N]
        A_q_all_power.append(A_q_m_power.unsqueeze(1))

    A_q_all_power = torch.cat(A_q_all_power, dim=1) # [M+1, M, N, N]
    qs = torch.tensor([j / 2 / (M+1) for j in range(0, M+1)]).to(edge_index.device)
    inverse_mat = torch.cat([torch.exp(-1j * 4 * torch.pi * k * qs).unsqueeze(0) for k in range(M+1)], dim=0).to(edge_index.device) # [M+1, M+1]
    inverse_mat = inverse_mat / (M+1)


    walk_profile = torch.einsum('kq, qlnm->klnm', inverse_mat, A_q_all_power).real.round() # [direction_idx, length_idx, N, N]


    # output
    if output=='profile':
        return [walk_profile[:m+2, m, :, :] for m in range(M)]
    elif output=='spd':
        idx = torch.arange(0, M).to(edge_index.device)
        wp = walk_profile[idx+1, idx,:, :]
        spd = torch.argmax((wp > 0.).float(), dim=0) + 1.
        mask = wp.sum(dim=0) == 0.
        spd[mask] = 0.
        return spd
        #spd_matrix = torch.ones_like(A) * torch.inf
        #for m in range(M):
        #    w = walk_profile[m][-1] # number of directed walks
        #    indices = torch.where(w > 0)
        #    spd_at_indices = spd_matrix[indices]
        #    replace_indices = torch.where(spd_at_indices > m+1)
        #    spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m+1
        #return spd_matrix
    elif output == 'lpd':
        lpd_matrix = torch.ones_like(A_q[0].real) * -1
        for m in reversed(range(M)):
            w = walk_profile[m][-1]  # number of directed walks
            indices = torch.where(w > 0)
            lpd_at_indices = lpd_matrix[indices]
            replace_indices = torch.where(lpd_at_indices < m + 1)
            lpd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        lpd_matrix[torch.where(lpd_matrix == -1)] = torch.inf
        return lpd_matrix
    elif output=='uspd':
        spd_matrix = torch.ones_like(A) * torch.inf
        for m in range(M):
            w = torch.max(walk_profile[m], dim=0)[0]  # number of undirected walks
            indices = torch.where(w > 0)
            spd_at_indices = spd_matrix[indices]
            replace_indices = torch.where(spd_at_indices > m + 1)
            spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        return spd_matrix


def get_walk_profile(edge_index, M, output='spd'):
    # input: (directed) edge index
    # output: walk profile matrix up to M steps

    # step 1: construct walk sequence (A_q, A^2_q, A^3_q, ..., A^M_q)
    edge_index = remove_self_loops(edge_index)[0]
    A = to_dense_adj(edge_index)[0]
    A_q = torch.tile(torch.unsqueeze(A, 0), (M+1, 1, 1))
    phase_factor = torch.exp(1j * 2 * torch.pi * torch.tensor([j / 2 / (M+1) for j in range(0, M+1)]))
    A_q = A_q * phase_factor.unsqueeze(-1).unsqueeze(-1) # [M+1, N, N]
    A_q = A_q + torch.conj(torch.transpose(A_q, 1, 2))
    A_q_m_power = torch.tile(torch.eye(A_q.size(-1), A_q.size(-1)).unsqueeze(0), (M+1, 1, 1)) # [M+1, N, N]
    A_q_m_power = torch.complex(A_q_m_power, torch.zeros_like(A_q_m_power))
    walk_profile = []
    for m in range(1, M+1):
        A_q_m_power = torch.bmm(A_q_m_power, A_q) * (phase_factor.unsqueeze(-1).unsqueeze(-1)) # [M+1, N, N]
        qs = torch.tensor([j / 2 / (M+1) for j in range(0, M+1)])
        inverse_mat = torch.cat([torch.exp(-1j * 4 * torch.pi * k * qs).unsqueeze(0) for k in range(m+1)], dim=0)
        inverse_mat = inverse_mat / (M+1)
        walk = torch.einsum('kq, qnm->knm', inverse_mat, A_q_m_power)
        walk_profile.append(walk.real.round())


    # output
    if output=='profile':
        return walk_profile
    elif output=='spd':
        walk_profile = torch.cat([wp[-1].unsqueeze(0) for wp in walk_profile], dim=0)
        spd = torch.argmax((walk_profile > 0.).float(), dim=0) + 1.
        mask = walk_profile.sum(dim=0) == 0.
        spd[mask] = 0.
        return spd
        #spd_matrix = torch.ones_like(A) * torch.inf
        #for m in range(M):
        #    w = walk_profile[m][-1] # number of directed walks
        #    indices = torch.where(w > 0)
        #    spd_at_indices = spd_matrix[indices]
        #    replace_indices = torch.where(spd_at_indices > m+1)
        #    spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m+1
        #return spd_matrix
    elif output == 'lpd':
        lpd_matrix = torch.ones_like(A_q[0].real) * -1
        for m in reversed(range(M)):
            w = walk_profile[m][-1]  # number of directed walks
            indices = torch.where(w > 0)
            lpd_at_indices = lpd_matrix[indices]
            replace_indices = torch.where(lpd_at_indices < m + 1)
            lpd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        lpd_matrix[torch.where(lpd_matrix == -1)] = torch.inf
        return lpd_matrix
    elif output=='uspd':
        spd_matrix = torch.ones_like(A) * torch.inf
        for m in range(M):
            w = torch.max(walk_profile[m], dim=0)[0]  # number of undirected walks
            indices = torch.where(w > 0)
            spd_at_indices = spd_matrix[indices]
            replace_indices = torch.where(spd_at_indices > m + 1)
            spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        return spd_matrix


def get_walk_profile_from_pe(pe, Lambda, degree, M, output='spd'):
    # pe: [Q, N, d], Lambda: [Q, d], d = N
    id = torch.eye(Lambda.size(-1))
    Lambda_diag = torch.einsum('qn, nm->qnm', Lambda, id)
    L_q = torch.einsum('qnk,qkl,qlm->qnm', pe, torch.complex(Lambda_diag, torch.zeros_like(Lambda_diag)), torch.conj(pe.transpose(1, 2)))
    degree_sqrt = torch.sqrt(degree.sum(-1))
    A_q = torch.diag(degree.sum(-1)).unsqueeze(0) - torch.einsum('n, qnm, m->qnm', degree_sqrt, L_q, degree_sqrt)

    phase_factor = torch.exp(1j * 2 * torch.pi * torch.tensor([j / 2 / (M+1) for j in range(1, M+2)]))
    A_q_m_power = torch.tile(torch.eye(A_q.size(-1), A_q.size(-1)).unsqueeze(0), (M + 1, 1, 1))  # [M+1, N, N]
    A_q_m_power = torch.complex(A_q_m_power, torch.zeros_like(A_q_m_power))
    walk_profile = []
    for m in range(1, M + 1):
        A_q_m_power = torch.bmm(A_q_m_power, A_q) * (phase_factor.unsqueeze(-1).unsqueeze(-1))  # [M+1, N, N]
        qs = torch.tensor([j / 2 / (M + 1) for j in range(1, M + 2)])
        inverse_mat = torch.cat([torch.exp(-1j * 4 * torch.pi * k * qs).unsqueeze(0) for k in range(m + 1)], dim=0)
        inverse_mat = inverse_mat / (M + 1)
        walk = torch.einsum('kq, qnm->knm', inverse_mat, A_q_m_power)
        walk_profile.append(walk.real.round())

    if output == 'profile':
        return walk_profile
    elif output == 'spd':
        spd_matrix = torch.ones_like(A_q[0].real) * torch.inf
        for m in range(M):
            w = walk_profile[m][-1]  # number of directed walks
            indices = torch.where(w > 0)
            spd_at_indices = spd_matrix[indices]
            replace_indices = torch.where(spd_at_indices > m + 1)
            spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        return spd_matrix
    elif output == 'lpd':
        lpd_matrix = torch.ones_like(A_q[0].real) * -1
        for m in reversed(range(M)):
            w = walk_profile[m][-1]  # number of directed walks
            indices = torch.where(w > 0)
            lpd_at_indices = lpd_matrix[indices]
            replace_indices = torch.where(lpd_at_indices < m + 1)
            lpd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        lpd_matrix[torch.where(lpd_matrix == -1)] =  torch.inf
        return lpd_matrix
    elif output == 'uspd':
        spd_matrix = torch.ones_like(A) * torch.inf
        for m in range(M):
            w = torch.max(walk_profile[m], dim=0)[0]  # number of undirected walks
            indices = torch.where(w > 0)
            spd_at_indices = spd_matrix[indices]
            replace_indices = torch.where(spd_at_indices > m + 1)
            spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        return spd_matrix



def check_eigenvectors(edge_index, eigvecs, eigenvals, pe_dim, q_dim, q):
    # eigvecs: [N, Q*pe_dim*2], eigenvals: [Q*pe_dim]
    #z = eigvecs.unflatten(-1, (q_dim, pe_dim, 2))
    z = eigvecs
    E = eigenvals # [Q, pe_dim]
    edge_index_new, edge_weight_list = get_mag_laplacian(
    edge_index,
    None,
    normalization='sym',
    num_nodes=eigvecs.size(0),
    q=q,
    dynamic_q=False,
    multiple_q=q_dim,
    )
    for i, edge_weight in enumerate(edge_weight_list):
        Lap = 1j* torch.zeros([eigvecs.size(0), eigvecs.size(0)])
        Lap[edge_index_new[0], edge_index_new[1]] = edge_weight
        assert (Lap == Lap.conj().T).all() # check Hermitian
        for j in range(pe_dim):
            z_new = Lap @ z[:, i, j]
            assert torch.abs(z_new - E[0, i, j] * z[:, i, j]).sum() < 1e-5




