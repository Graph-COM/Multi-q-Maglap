import os
from typing import List, Sequence, Tuple, Union

from absl import app
from absl import flags
from absl import logging
#import jraph
#import numba
import numpy as np
import tqdm
from torch_geometric.data import Data
import torch
from copy import deepcopy
import pickle

np.random.seed(42)

_OUT_PATH = flags.DEFINE_string('out_path', '../data/sorting',
                                'The path to write datasets to.')
_SHARD_SIZE = flags.DEFINE_integer(
    'shard_size', 10_000, 'The number of times to store in each file.')
_REVERSE = flags.DEFINE_boolean('reverse', True,
                                'If true also reverse test samples')

# Instructions with lengths to put in which dataset and how many sorting
# networks shall be generated (2x for train, 3x for valid and test)
SPLITS = [
    # (split, seq_lens, n_generation_trials)
    ('train', [7, 8, 9, 10, 11], 400_000),
    #('train', [7, 8, 9, 10, 11], 500),
    ('valid', [12], 20_000),
    #('valid', [12], 500),
    ('test', [13, 14, 15, 16], 20_000)
    #('test', [13, 14, 15, 16], 500)
]

DATASET_NAME = '7to11_12_13to16'
# How many random sorting networks are sampled in parallel (multithreaded)
BATCH_SIZE = 500
# Upper limit on operators
MAX_OPERATORS = 512

maglap_configs = [
    dict(k=25, k_excl=0, q=0.25,
         q_absolute=False, norm_comps_sep=False,
         sign_rotate=True, use_symmetric_norm=True),
    dict(k=25, k_excl=0, q=0,
         q_absolute=False, norm_comps_sep=False,
         sign_rotate=True, use_symmetric_norm=True)
]

#@numba.jit(nopython=False)
def get_test_cases(seq_len: int) -> np.ndarray:
    """Generates all possible 0-1 perturbations (aka truth table)."""
    true_false = np.array((True, False))
    return np.stack(np.meshgrid(*((true_false,) * seq_len))).reshape(seq_len, -1)


#@numba.jit(nopython=False)
def generate_sorting_network(seq_len: int, max_operators: int = 512):
    """Generates a valid sorting network."""
    test_cases = get_test_cases(seq_len)
    operators = []
    last_operator = (-1, -1)
    unsorted_locations = np.arange(seq_len)
    while True:
        i, j = sorted(np.random.choice(unsorted_locations, size=2, replace=False))
        if (i, j) == last_operator:
            continue
        if i not in unsorted_locations and j not in unsorted_locations:
            continue
        last_operator = (i, j)
        operators.append((i, j))

        test_cases[(i, j), :] = np.sort(test_cases[(i, j), :], axis=0)
        test_cases = np.unique(test_cases, axis=1)

        unsorted_locations_ = np.arange(seq_len)[(np.sort(test_cases, axis=0) !=
                                                  test_cases).any(1)]
        # Sometime numba has issues with tracking variables through loops
        unsorted_locations = unsorted_locations_

        if test_cases.shape[1] == seq_len + 1:
            return True, operators, test_cases
        if len(operators) == max_operators:
            return False, operators, test_cases


#@numba.jit(nopython=False)
def test_network(operators, seq_len):
    test_cases = get_test_cases(seq_len)
    for i, j in operators.astype(np.int64):
        test_cases[(i, j), :] = np.sort(test_cases[(i, j), :], axis=0)
        test_cases = np.unique(test_cases, axis=1)
        if test_cases.shape[1] == seq_len + 1:
            return True, test_cases
    return False, test_cases


def operators_to_graphstuple(operators: Union[List[Tuple[int, int]],
                                              np.ndarray],
                             seq_len: int) -> Data:
    """Converts the list of "operators" to a jraph.graphstuple."""
    num_nodes = len(operators)
    senders = np.zeros(int(2 * len(operators)) - seq_len, dtype=np.int32)
    receivers = np.zeros(int(2 * len(operators)) - seq_len, dtype=np.int32)
    # Node features: (order_idx, location_i, location_j)
    nodes = np.zeros((num_nodes, 3), dtype=np.float32)

    loc = {i: -1 for i in range(seq_len)}
    edge_idx = 0
    for idx, (i, j) in enumerate(operators):
        # Add edges
        if loc[i] >= 0:
            senders[edge_idx] = loc[i]
            receivers[edge_idx] = idx
            edge_idx += 1
        if loc[j] >= 0:
            senders[edge_idx] = loc[j]
            receivers[edge_idx] = idx
            edge_idx += 1
        # Update node features
        nodes[idx, 0] = idx
        nodes[idx, 1] = i
        nodes[idx, 2] = j
        # Update mapping from location to node
        loc[i] = idx
        loc[j] = idx

    # change this to pyg graph
    edge_index = np.concatenate((senders[:, None], receivers[:, None]), axis=-1).T
    return Data(x=torch.tensor(nodes), edge_index=torch.tensor(edge_index), num_nodes=num_nodes)
    #return jraph.GraphsTuple(
    #    n_node=np.array([num_nodes], dtype=np.int32),
    #    n_edge=np.array(senders.shape, dtype=np.int32),
    #    senders=senders,
    #    receivers=receivers,
    #    nodes=dict(node_feat=nodes),
    #    edges=np.array([], dtype=np.float32),
    #    globals=np.array([], dtype=np.float32))


def generate_sorting_network_batch(
        seq_len: int,
        batch: int,
        max_operators: int = 512
) -> List[Data]:
    """Generates batch graphs in parallel."""
    graph_tuples = []
    #for _ in numba.prange(batch):
    for _ in range(batch):
        success, operators, _ = generate_sorting_network(
            seq_len, max_operators=max_operators)
        if success:
            graph_tuples.append(operators_to_graphstuple(operators, seq_len))
            #graph_tuples[-1] = precalc_and_append(graph_tuples[-1], maglap_configs) # no need to get maglap for now
    return graph_tuples


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    base_path = os.path.join(_OUT_PATH.value, DATASET_NAME)
    os.makedirs(base_path, exist_ok=True)

    id_ = 0
    for split, seq_lens, n_generation_trials in SPLITS:
        file_path = os.path.join(base_path, split)
        os.makedirs(file_path, exist_ok=True)

        sample_count = 0
        buffer = []
        start_id = id_
        n_batches = n_generation_trials // BATCH_SIZE
        for batch_idx in tqdm.tqdm(range(n_batches), desc=split):
            seq_len = np.random.choice(seq_lens, 1).item()
            graphs = generate_sorting_network_batch(
                seq_len,
                BATCH_SIZE,
                MAX_OPERATORS)

            sample_count += (3 if _REVERSE.value and split != 'train' else
                             2) * len(graphs)

            for graph in graphs:
                #buffer.append(
                    #(graph, np.array([True]), np.array([True]), np.array(id_)))
                graph.y = torch.tensor(1.)
                buffer.append(graph)
                id_ += 1
                # Remove last operation to generate an incorrect sorting network
                # change it to pyg graph
                #graph_ = jraph.GraphsTuple(
                #    nodes=dict(node_feat=graph.nodes['node_feat'][:-1]),
                #    edges=np.array([], dtype=np.float32),
                #    # It is very unlikely that last operation still operates on inputs
                #    senders=graph.senders[:-2],
                #    receivers=graph.receivers[:-2],
                #    n_node=graph.n_node - 1,
                #    n_edge=graph.n_edge - 2,
                #    globals=np.array([], dtype=np.float32))
                graph_ = Data(x=graph.x[:-1], edge_index=graph.edge_index[:, :-2], num_nodes=graph.num_nodes-1,
                              y=torch.tensor(0.))
                buffer.append(graph_)
                #buffer.append(
                #    (graph_, np.array([False]), np.array([False]), np.array(id_)))
                id_ += 1

                if _REVERSE.value and split != 'train':
                    #operators = graph.nodes['node_feat'][::-1, 1:]
                    operators = np.array(torch.flip(graph.x, dims=[0,])[:, 1:])
                    is_correct, _ = test_network(operators, seq_len)
                    graph_ = operators_to_graphstuple(operators, seq_len)
                    graph_.y = torch.tensor(is_correct).float()
                    #buffer.append((graph_, np.array([is_correct]),
                                   #np.array([is_correct]), np.array(id_)))
                    buffer.append(graph_)
                    id_ += 1

            if len(buffer) >= _SHARD_SIZE.value or batch_idx == n_batches - 1:
                #file_name = os.path.join(file_path, f'{start_id}_{id_ - 1}.npz')
                #np.savez_compressed(file_name, data=np.array(buffer, dtype='object'))
                file_name = os.path.join(file_path, f'{start_id}_{id_ - 1}.pkl')
                with open(file_name, 'wb') as f:
                    pickle.dump(buffer, f)
                logging.info('Wrote %d to %s', len(buffer), file_name)
                buffer = []
                start_id = id_

        logging.info('Wrote %d instances in `%s`', sample_count, split)


if __name__ == '__main__':
    app.run(main)
