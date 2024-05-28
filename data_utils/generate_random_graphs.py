import sys
sys.path.append('../')
import numpy as np
import os
import torch
import torch_geometric as pyg
import networkx as nx
from absl import flags
from absl import logging
from absl import app
from typing import Sequence, Tuple
import tqdm
from calculate_distance import longest_path_distance
from utils.get_mag_lap import get_walk_profile


np.random.seed(42)

_OUT_PATH = flags.DEFINE_string('out_path', '../data/distance',
                                'The path to write datasets to.')
_SHARD_SIZE = flags.DEFINE_integer(
  'shard_size', 10_000, 'The number of times to store in each file.')
_WEIGHTED = flags.DEFINE_bool('weighted', False,
                              'If the dataset should contain weighted graphs.')
_CONNECTED = flags.DEFINE_bool(
  'connected', True,
  'If the dataset should contain only (weakly) connected graphs.')
_ACYCLIC = flags.DEFINE_bool(
  'acyclic', False,
  'If the dataset should contain only acyclic graphs.')
_TARGET = flags.DEFINE_enum(
  'target', 'directed', ['directed', 'undirected', 'signed'],
  'How the distance should be calculated.')
_N_TRAIN = flags.DEFINE_list(
  'n_train', [16, 18], 'Range `(min, max+1)` of number of nodes for train')
_N_VALID = flags.DEFINE_list(
  'n_valid', [18, 20], 'Range `(min, max+1)` of number of nodes for validation')
_N_TEST = flags.DEFINE_list(
  'n_test', [20, 28], 'Range `(min, max+1)` of number of nodes for test')

DIST = 'spd' # 'spd' or 'lpd' or 'wp'


maglap_configs = [
  dict(k=16, k_excl=0, q=0.1,
       q_absolute=False, norm_comps_sep=False,
       sign_rotate=True, use_symmetric_norm=True),
  dict(k=16, k_excl=0, q=0,
       q_absolute=False, norm_comps_sep=False,
       sign_rotate=True, use_symmetric_norm=True)
]


AVERAGE_DEGREE = {
  'acyclic': (1, 1.5, 2, 2.5, 3),
  'regular': (1, 1.5, 2)
}



def _random_er_graph(nb_nodes, p=0.5, directed=True, acyclic=False,
                     weighted=False, low=0.1, high=1.0):
  """Random Erdos-Renyi graph."""

  mat = np.random.binomial(1, p, size=(nb_nodes, nb_nodes))
  if not directed:
    mat *= np.transpose(mat)
  elif acyclic:
    mat = np.triu(mat, k=1)
    p = np.random.permutation(nb_nodes)  # To allow nontrivial solutions
    mat = mat[p, :][:, p]
  if weighted:
    weights = np.random.uniform(low=low, high=high, size=(nb_nodes, nb_nodes))
    if not directed:
      weights *= np.transpose(weights)
      weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
    mat = mat.astype(float) * weights
  return mat


def generate_sample(**random_er_graph_kwargs):
  adj = _random_er_graph(**random_er_graph_kwargs)
  G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
  if _CONNECTED.value:
    G = G.subgraph(max(nx.weakly_connected_components(G), key=len))
    adj = nx.to_numpy_array(G)

  if _TARGET.value == 'undirected':
    G = G.to_undirected()
  elif _TARGET.value == 'signed':
    adj_signed = adj[:]
    adj_signed -= ~adj_signed.astype(bool) & adj_signed.T
    G = nx.from_numpy_array(adj_signed, create_using=nx.DiGraph)
  # distances = np.concatenate((np.expand_dims(spd_distances, -1), np.expand_dims(lpd_distances, -1)), -1)


  senders, receivers = np.where(adj)
  #senders, receivers = torch.tensor(senders), torch.tensor(receivers)
  edge_index = np.concatenate([senders.reshape([1, -1]), receivers.reshape([1, -1])], axis=0)
  #graph = pyg.data.Data(edge_index=edge_index, num_nodes=adj.shape[0], y=distances)

  if DIST == 'spd':
    distances = nx.floyd_warshall_numpy(G)
  elif DIST == 'lpd':
    distances = longest_path_distance(G)
  elif DIST == 'wp':
    distances = get_walk_profile(torch.tensor(edge_index), M=4, output='profile')
    distances = distances[-1] # walk profile at step 4
    distances = distances.transpose(0, -1).transpose(0, 1)
    distances = np.array(distances)

  return edge_index, distances


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  assert len(_N_TRAIN.value) == 2, '`n_train` must be of length 2'
  assert len(_N_VALID.value) == 2, '`n_valid` must be of length 2'
  assert len(_N_TEST.value) == 2, '`n_test` must be of length 2'
  n_train = tuple(int(v) for v in _N_TRAIN.value)
  n_valid = tuple(int(v) for v in _N_VALID.value)
  n_test = tuple(int(v) for v in _N_TEST.value)

  # Instructions with lengths to put in which dataset and how many sorting
  # networks shall be generated (excluding random sampling over topo. orders)
  splits = [
    # (split, nb_nodes_list, n_generation_trials)
    ('train', list(range(*n_train)), 400_000), # training & validation set
    #('train', list(range(*n_train)), 10),
    #('train', list(range(*n_train)), 4_000),
    ('valid', list(range(*n_valid)), 5_000), # this will be the test set
    #('valid', list(range(*n_valid)), 10),
    #('test', list(range(*n_test)), 2_000)
    ('test', list(range(*n_test)), 10) # not going to use this
  ]

  dataset_name = f'{n_train[0]}to{n_train[1] - 1}_{n_valid[0]}to{n_valid[1] - 1}_{n_test[0]}to{n_test[1] - 1}'
  if _WEIGHTED.value or _CONNECTED.value or _ACYCLIC.value:
    dataset_name += '_'
    if _WEIGHTED.value:
      dataset_name += 'w'
    if _CONNECTED.value:
      dataset_name += 'c'
    if _ACYCLIC.value:
      dataset_name += 'a'
    if _TARGET.value == 'undirected':
      dataset_name += '_u'
    elif _TARGET.value == 'signed':
      dataset_name += '_s'
  if DIST == 'lpd':
    dataset_name += '_lpd'
  elif DIST == 'wp':
    dataset_name += '_wp'


  base_path = os.path.join(_OUT_PATH.value, dataset_name, 'raw')
  os.makedirs(base_path, exist_ok=True)

  average_degree = (
    AVERAGE_DEGREE['acyclic'] if _ACYCLIC.value else AVERAGE_DEGREE['regular'])

  id_ = 0
  for split, nb_nodes_list, n_generation_trials in splits:
    file_path = os.path.join(base_path, split)
    os.makedirs(file_path, exist_ok=True)

    sample_count = 0
    buffer = []
    start_id = id_
    for trial in tqdm.tqdm(range(n_generation_trials), desc=split):
      nb_nodes = np.random.choice(nb_nodes_list, 1).item()
      deg = np.random.choice(average_degree, 1).item()
      p = deg / (nb_nodes - 1)
      graph, distances = generate_sample(
        nb_nodes=nb_nodes, p=p, directed=True, acyclic=_ACYCLIC.value,
        weighted=_WEIGHTED.value)

      sample_count += 1
      buffer.append((graph, distances, np.array(id_)))
      #buffer.append((
      #  graph, distances.flatten(), distances.flatten(), np.array(id_)))
      id_ += 1

      if len(buffer) >= _SHARD_SIZE.value or trial == n_generation_trials - 1:
        file_name = os.path.join(file_path, f'{start_id}_{id_ - 1}.npz')
        np.savez_compressed(file_name, data=np.array(buffer, dtype='object'))
        logging.info('Wrote %d to %s', len(buffer), file_name)
        buffer = []
        start_id = id_

    logging.info('Wrote %d instances in `%s`', sample_count, split)



if __name__ == '__main__':
  app.run(main)
