import networkx as nx
import numpy as np
def single_source_longest_dag_path_length(graph, s):
    dist = dict.fromkeys(graph.nodes, -float('inf'))
    dist[s] = 0.
    topo_order = nx.topological_sort(graph)
    for n in topo_order:
        for s in graph.successors(n):
            if dist[s] < dist[n] + 1:
                dist[s] = dist[n] + 1
    return dist

def longest_path_distance_dag(G):
    num_nodes = len(G)
    dist = np.zeros([num_nodes, num_nodes])
    for i, node in enumerate(G.nodes):
        d_from_node = single_source_longest_dag_path_length(G, node)
        dist[i] = np.array(list(d_from_node.values()))
    dist[np.where(dist == - np.inf)] = np.inf
    return dist


def longest_path_distance(G):
    num_nodes = len(G)
    dist = np.zeros([num_nodes, num_nodes])
    for i, s in enumerate(G.nodes):
        for j, t in enumerate(G.nodes):
            paths = nx.all_simple_paths(G, s, t, cutoff=10)
            lpd = 0
            for p in list(paths):
                if len(p) - 1 > lpd:
                    lpd = len(p) - 1
            dist[i, j] = lpd
    return dist


