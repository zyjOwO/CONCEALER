from typing import Iterator

import torch
from torch_geometric.data import Data
from tqdm import tqdm


def get_concealer_size(graph: Data):
    graph.label = graph.concealer_mask.sum() - 1
    graph = graph.subgraph(~graph.concealer_mask.bool())
    return graph.sort()


def get_binding_sites(graph: Data, contain_concealer=False):
    concealer_nids = graph.concealer_mask.nonzero().view(-1)
    src_indices, dst_indices = graph.edge_index
    src_mask = torch.isin(src_indices, concealer_nids).float()
    dst_mask = torch.isin(dst_indices, concealer_nids).float()
    target_mask = src_mask + dst_mask
    target_indices = torch.where(target_mask == 1.)[0]

    binding_sites = torch.zeros(graph.num_nodes)
    concealer_nids = set(concealer_nids.tolist())
    src_indices = src_indices.tolist()
    dst_indices = dst_indices.tolist()
    for idx in target_indices:
        if not contain_concealer:
            edge = set([src_indices[idx], dst_indices[idx]])
            nid = list(edge - concealer_nids)[0]
            binding_sites[nid] += 1
        else:
            binding_sites[src_indices[idx]] += 1
            binding_sites[dst_indices[idx]] += 1
    label = binding_sites > .5
    graph.label = label.to(torch.float32)

    if not contain_concealer:
        graph = graph.subgraph(~graph.concealer_mask.bool())
    else:
        edge_mask = torch.ones(graph.num_edges, dtype=torch.bool)
        edge_mask[target_indices] = False
        graph.edge_index = graph.edge_index[:, edge_mask]
        graph.edge_attr = graph.edge_attr[edge_mask]
    return graph.sort()


def compute_stats(graphs: Iterator):
    stats = {
        'max_concealer_size': {},
        'concealer_size_weight': {},
        'binding_sites_weight': {},
        'non_binding_sites_weight': {},
    }
    for graph in tqdm(graphs, leave=False, desc='computing statistics'):
        concealer_size_prob = torch.zeros(1000, dtype=torch.float32)
        binding_sites_prob = torch.zeros(1, dtype=torch.float32)
        num_masks = 0

        for mask in graph.concealer_masks:
            num_masks += 1
            concealer_size = sum(mask)
            assert concealer_size > 0
            concealer_size_prob[concealer_size - 1] += 1
            num_nodes_without_concealers = graph.num_nodes - concealer_size

            concealer_nids = mask.nonzero().view(-1)
            src_indices, dst_indices = graph.edge_index
            src_mask = torch.isin(src_indices, concealer_nids).float()
            dst_mask = torch.isin(dst_indices, concealer_nids).float()
            target_mask = src_mask + dst_mask
            target_indices = torch.where(target_mask == 1.)[0]

            binding_sites = set()
            concealer_nids = set(concealer_nids.tolist())
            src_indices = src_indices.tolist()
            dst_indices = dst_indices.tolist()
            for idx in target_indices:
                binding_sites |= set([src_indices[idx], dst_indices[idx]])
            binding_sites -= concealer_nids

            num_binding_sites = len(binding_sites)
            binding_sites_prob += (num_binding_sites / num_nodes_without_concealers)

        max_concealer_size = max(torch.where(concealer_size_prob > 0.)[0]) + 1
        concealer_size_prob = concealer_size_prob[:max_concealer_size]
        concealer_size_prob /= num_masks
        binding_sites_prob /= num_masks

        concealer_size_weight = (1 / concealer_size_prob) / max_concealer_size
        binding_sites_weight = 1 / binding_sites_prob
        non_binding_sites_weight = 1 / (1 - binding_sites_prob)

        concealer_size_weight[concealer_size_weight == torch.inf] = 1e-12
        binding_sites_weight[binding_sites_weight == torch.inf] = 1e-12
        non_binding_sites_weight[non_binding_sites_weight == torch.inf] = 1e-12

        stats['max_concealer_size'][graph.graph_name] = max_concealer_size
        stats['concealer_size_weight'][graph.graph_name] = concealer_size_weight
        stats['binding_sites_weight'][graph.graph_name] = binding_sites_weight
        stats['non_binding_sites_weight'][graph.graph_name] = non_binding_sites_weight
    return stats
