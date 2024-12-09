# Dataset statistics
from pathlib import Path

import numpy as np

from blosc_compress import blosc_pkl_load


def stats(dataset_name):
    processed_dir = Path(f'dataset/{dataset_name}/processed')
    benign_dir = processed_dir.joinpath('benign')
    malicious_dir = processed_dir.joinpath('anomaly')
    n_nodes = []
    n_edges = []
    n_benign_graphs = np.zeros((1,), dtype=np.int64)[0]
    n_malicious_graphs = np.zeros((1,), dtype=np.int64)[0]
    for graph_path in benign_dir.glob('*.blosc'):
        nx_graph = blosc_pkl_load(graph_path)
        n_nodes.append(nx_graph.number_of_nodes())
        n_edges.append(nx_graph.number_of_edges())
        n_benign_graphs += 1
    for graph_path in malicious_dir.glob('*.blosc'):
        nx_graph = blosc_pkl_load(graph_path)
        n_nodes.append(nx_graph.number_of_nodes())
        n_edges.append(nx_graph.number_of_edges())
        n_malicious_graphs += 1
    print(f'n_nodes: {sum(n_nodes)}; n_edges: {sum(n_edges)}; n_benign_graphs: {n_benign_graphs}; n_malicious_graphs: {n_malicious_graphs}')
    return


if __name__ == '__main__':
    stats('supply')
