import torch
from torch_geometric.utils import remove_self_loops

from torch_geometric.utils import (
    to_dense_adj
)

def naive_walk_profile_calculator(edge_index, M):
    edge_index = remove_self_loops(edge_index)[0]
    A = to_dense_adj(edge_index)[0]
    all_walks = naive_walk_profile_recursive(A, M)
    return all_walks


def naive_walk_profile_recursive(A, m):
    # compute from naive_walk_profile_recursive(A, m-1)
    if m < 1:
        raise Exception("invalid m")
    if m == 1:
        return {'0': A.T, '1': A}
    else:
        prev = naive_walk_profile_recursive(A, m-1)
        new_res = {}
        for key in prev:
            new_res[key+'0'] = prev[key] @ A.T
            new_res[key + '1'] = prev[key] @ A
        return new_res
