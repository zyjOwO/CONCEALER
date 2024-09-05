from itertools import product

import numpy as np
import scipy.sparse as sp
import torch
from numpy.random import standard_normal
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index, to_scipy_sparse_matrix
from tqdm import tqdm

from .noise_schedule import PredefinedNoiseSchedule, sample_gaussian_with_mask


def get_components(data: Data, connection='weak'):
    assert data.edge_index is not None
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    _, component = sp.csgraph.connected_components(adj, connection=connection)
    _, count = np.unique(component, return_counts=True)
    largest_subset = np.in1d(component, count.argsort()[-1:])
    small_subsets = (~largest_subset).nonzero()[0]
    return largest_subset.nonzero()[0], small_subsets


class AtlasApplyNoise():
    pass


def apply_noise(graph: Data, diffusion_steps=None, noise_schedule=None, binding_sites='neighbor'):
    r'''adding Gaussian noise
    '''
    graph = graph.clone()
    x = graph.x.to(torch.float32)  # (num_nodes, num_node_features)
    edge_index = graph.edge_index.to(torch.int64)  # (2, num_edges)
    edge_attr = graph.edge_attr.to(torch.float32)  # (num_edges, num_edge_features)
    concealer_mask = graph.concealer_mask.to(torch.bool)  # (num_nodes)

    mask_indices = concealer_mask.nonzero().view(-1).numpy()
    edge_index = edge_index.t().numpy()
    edge_attr = edge_attr.numpy()

    if binding_sites == 'neighbor':
        src_indices = edge_index[:, 0]
        dst_indices = edge_index[:, 1]
        src_mask = np.isin(src_indices, mask_indices).astype(np.float32)
        dst_mask = np.isin(dst_indices, mask_indices).astype(np.float32)
        target_mask = src_mask + dst_mask
        target_indices = np.where(target_mask == 1.)[0]
        contaminated_nodes = set()
        src_indices = src_indices.tolist()
        dst_indices = dst_indices.tolist()
        for idx in target_indices:
            contaminated_nodes |= set([src_indices[idx], dst_indices[idx]])
        contaminated_nodes -= set(mask_indices)
        binding_sites = contaminated_nodes
    elif isinstance(binding_sites, Tensor):
        binding_sites = set(binding_sites.nonzero().view(-1).tolist())
    else:
        raise ValueError(binding_sites)

    # noisy concealer edges (new_edges)
    mask_indices = set(mask_indices.tolist())
    new_edges = set(product(mask_indices, mask_indices))  # concealer <-> concealer
    new_edges |= set(product(mask_indices, binding_sites))  # concealer -> binding_sites
    new_edges |= set(product(binding_sites, mask_indices))  # concealer <- binding_sites
    # discard original edges
    new_edges -= set(map(tuple, edge_index.tolist()))
    # discard self-loops
    new_edges = np.array(list(new_edges)).T
    mask = new_edges[0] != new_edges[1]
    new_edges = new_edges[:, mask].T
    # add new_edges into edge_index
    edge_index = np.vstack((edge_index, new_edges))

    # the feature of new_edges (new_edge_attr)
    if diffusion_steps is not None and noise_schedule is not None:
        new_edges_attr = np.zeros((new_edges.shape[0], edge_attr.shape[1]), dtype=np.float32)
        new_edges_attr[:, 0] = 1.  # [1,0,0,...] non-edge
    else:
        new_edges_attr = standard_normal((new_edges.shape[0], edge_attr.shape[1])).astype(np.float32)  # Gaussian noise
    # add new_edge_attr into edge_attr
    edge_attr = np.vstack((edge_attr, new_edges_attr))
    edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)

    # generate noisy concealer component
    mask_indices = np.array(list(mask_indices))
    if diffusion_steps is not None and noise_schedule is not None:  # training on benign graphs with concealer_masks
        # sample t
        node_t_array = torch.randint(0, diffusion_steps + 1, size=(x.shape[0], 1)).float()
        node_t_array = node_t_array / diffusion_steps
        edge_t_array = torch.randint(0, diffusion_steps + 1, size=(edge_attr.shape[0], 1)).float()
        edge_t_array = edge_t_array / diffusion_steps
        # alpha_t and sigma_t
        noise_schedule = PredefinedNoiseSchedule(noise_schedule, diffusion_steps)
        node_alpha_t = noise_schedule.get_alpha(t_normalized=node_t_array)
        node_sigma_t = noise_schedule.get_sigma(t_normalized=node_t_array)
        edge_alpha_t = noise_schedule.get_alpha(t_normalized=edge_t_array)
        edge_sigma_t = noise_schedule.get_sigma(t_normalized=edge_t_array)
        # add noise
        concealer_node_mask = concealer_mask.unsqueeze(-1)  # (num_nodes, 1)
        eps_x = sample_gaussian_with_mask(size=x.shape, mask=concealer_node_mask)
        concealer_edge_mask = np.isin(edge_index, mask_indices).any(axis=1)
        concealer_edge_mask = torch.as_tensor(concealer_edge_mask, dtype=torch.bool).unsqueeze(-1)
        eps_edge_attr = sample_gaussian_with_mask(size=edge_attr.shape, mask=concealer_edge_mask)
        # z_t ~ q(z_t|x,e) = {x_t, e_t}
        # x_t
        x_t = node_alpha_t * x + node_sigma_t * eps_x
        x_t = x * ~concealer_node_mask + x_t * concealer_node_mask
        # e_t (edge_attr_t)
        e_t = edge_alpha_t * edge_attr + edge_sigma_t * eps_edge_attr
        e_t = edge_attr * ~concealer_edge_mask + e_t * concealer_edge_mask
    else:  # testing on malicious graphs
        x_t = x
        e_t = edge_attr
        concealer_node_mask = concealer_mask.unsqueeze(-1)
        concealer_edge_mask = np.isin(edge_index, mask_indices).any(axis=1)
        concealer_edge_mask = torch.as_tensor(concealer_edge_mask, dtype=torch.bool).unsqueeze(-1)

    edge_index = torch.as_tensor(edge_index, dtype=torch.int64).t()
    sorted_edge_index, [sorted_edge_attr, sorted_e_t,
                        concealer_edge_mask] = sort_edge_index(edge_index=edge_index,
                                                               edge_attr=[edge_attr, e_t, concealer_edge_mask],
                                                               num_nodes=graph.num_nodes,
                                                               sort_by_row=True)
    noisy_graph = Data(x=x_t, edge_index=sorted_edge_index, edge_attr=sorted_e_t, graph_name=graph.graph_name)
    noisy_graph.concealer_node_mask = concealer_node_mask
    noisy_graph.concealer_edge_mask = concealer_edge_mask
    clean_graph = Data(x=x, edge_index=sorted_edge_index, edge_attr=sorted_edge_attr, graph_name=graph.graph_name)
    clean_graph.concealer_node_mask = concealer_node_mask
    clean_graph.concealer_edge_mask = concealer_edge_mask
    return noisy_graph, clean_graph


def compute_stats(graphs, ntype_list, etype_list):
    stats = {
        'concealer_size_prob': {},
        'concealer_ntype_prob': {},
        'concealer_etype_prob': {},
        'concealer_size_weight': {},
        'concealer_ntype_weight': {},
        'concealer_etype_weight': {},
        'node_prob': {},
        'ntype_prob': {},
        'etype_prob': {},
        'node_weight': {},
        'ntype_weight': {},
        'etype_weight': {}
    }
    max_num_nodes = 10000
    # p(graph_size) i.e., p(num_nodes)
    node_prob = torch.zeros(max_num_nodes, dtype=torch.float32)
    # p(ntype)
    ntype_prob = torch.zeros(len(ntype_list), dtype=torch.float32)
    # p(etype)
    etype_prob = torch.zeros(len(etype_list) + 1, dtype=torch.float32)

    for graph in tqdm(graphs, leave=False, desc='computing statistical data'):
        # for whole dataset
        node_prob[graph.num_nodes - 1] += 1
        ntype_label = torch.argmax(graph.x.to(torch.int32), dim=1)
        for idx in range(len(ntype_list)):
            ntype_prob[idx] += (sum(ntype_label == idx) / graph.num_nodes)
        max_possible_num_edges = graph.num_nodes**2 - graph.num_nodes
        etype_prob[0] += (max_possible_num_edges - graph.num_edges) / max_possible_num_edges
        edge_label = torch.argmax(graph.edge_attr.to(torch.int16), dim=1)
        for idx in range(len(etype_list)):
            idx = idx + 1
            etype_prob[idx] += (sum(edge_label == idx) / max_possible_num_edges)

        # for each graph in the dataset
        concealer_size_prob = torch.zeros(1000, dtype=torch.float32)
        concealer_ntype_prob = torch.zeros(len(ntype_list), dtype=torch.float32)
        concealer_etype_prob = torch.zeros(len(etype_list) + 1, dtype=torch.float32)
        for mask in graph.concealer_masks:
            concealer_size = sum(mask)
            concealer_size_prob[concealer_size - 1] += 1
            current_ntype_label = torch.argmax(graph.x[mask].to(torch.int32), dim=1)
            for idx in range(len(ntype_list)):
                concealer_ntype_prob[idx] += (sum(current_ntype_label == idx) / concealer_size)
            concealer_nids = mask.nonzero().view(-1)
            src_indices, dst_indices = graph.edge_index
            src_mask = torch.isin(src_indices, concealer_nids).float()
            dst_mask = torch.isin(dst_indices, concealer_nids).float()
            target_mask = src_mask + dst_mask
            target_indices = torch.where(target_mask > .1)[0]
            binding_sites = set()
            edge_index_t = graph.edge_index.t().tolist()
            for idx in torch.where(target_mask == 1.)[0]:
                binding_sites |= set(edge_index_t[idx])
            binding_sites -= set(concealer_nids.tolist())
            num_binding_sites = len(binding_sites)
            max_possible_num_edges = (concealer_size**2 - concealer_size) + (num_binding_sites * concealer_size * 2)
            concealer_etype_prob[0] += (max_possible_num_edges - len(target_indices)) / max_possible_num_edges
            current_edge_label = torch.argmax(graph.edge_attr[target_indices].to(torch.int32), dim=1)
            for idx in range(len(etype_list)):
                idx = idx + 1
                concealer_etype_prob[idx] += (sum(current_edge_label == idx) / max_possible_num_edges)

        max_concealer_size = max(torch.where(concealer_size_prob > 0.)[0]) + 1
        concealer_size_prob = concealer_size_prob[:max_concealer_size]
        concealer_size_prob /= len(graph.concealer_masks)
        concealer_ntype_prob /= len(graph.concealer_masks)
        concealer_etype_prob /= len(graph.concealer_masks)
        concealer_size_weight = (1 / concealer_size_prob) / max_concealer_size
        concealer_ntype_weight = (1 / concealer_ntype_prob) / max_concealer_size
        concealer_etype_weight = (1 / concealer_etype_prob) / max_concealer_size

        concealer_size_weight[concealer_size_weight == torch.inf] = 0.
        concealer_ntype_weight[concealer_ntype_weight == torch.inf] = 0.
        concealer_etype_weight[concealer_etype_weight == torch.inf] = 0.

        stats['concealer_size_prob'][graph.graph_name] = concealer_size_prob
        stats['concealer_ntype_prob'][graph.graph_name] = concealer_ntype_prob
        stats['concealer_etype_prob'][graph.graph_name] = concealer_etype_prob
        stats['concealer_size_weight'][graph.graph_name] = concealer_size_weight
        stats['concealer_ntype_weight'][graph.graph_name] = concealer_ntype_weight
        stats['concealer_etype_weight'][graph.graph_name] = concealer_etype_weight

    max_num_nodes = max(torch.where(node_prob > 0.)[0]) + 1
    node_prob = node_prob[:max_num_nodes]
    node_prob /= node_prob.sum()
    ntype_prob /= ntype_prob.sum()
    etype_prob /= etype_prob.sum()
    node_weight = (1 / node_prob) / node_prob.shape[0]
    ntype_weight = (1 / ntype_prob) / ntype_prob.shape[0]
    etype_weight = (1 / etype_prob) / etype_prob.shape[0]

    node_weight[node_weight == torch.inf] = 0.
    ntype_weight[ntype_weight == torch.inf] = 0.
    etype_weight[etype_weight == torch.inf] = 0.

    stats['node_prob'] = node_prob
    stats['ntype_prob'] = ntype_prob
    stats['etype_prob'] = etype_prob
    stats['node_weight'] = node_weight
    stats['ntype_weight'] = ntype_weight
    stats['etype_weight'] = etype_weight
    return stats


def add_context_reverse_edges(graph: Data):
    graph = graph.clone()
    if hasattr(graph, 'concealer_edge_mask'):
        context_edge_mask = ~graph.concealer_edge_mask.view(-1)
    else:
        context_edge_mask = torch.ones((graph.num_edges,), dtype=torch.bool)
    edge_index = graph.edge_index.t()[context_edge_mask]
    edge_index_set = set(map(tuple, edge_index.tolist()))
    flipped_edge_index_set = set(map(tuple, torch.fliplr(edge_index).tolist()))
    new_edge_index_set = flipped_edge_index_set - edge_index_set
    new_edge_index = torch.tensor(list(new_edge_index_set), dtype=torch.int64).t()
    new_edge_attr = torch.zeros((new_edge_index.shape[1], graph.edge_attr.shape[1]), dtype=torch.float32)
    new_edge_attr[:, 0] = 1.
    for store in graph.edge_stores:
        if 'edge_index' not in store:
            continue
        keys, values = [], []
        for key, value in store.items():
            if store.is_edge_attr(key):
                if key == 'edge_index':
                    value = torch.hstack((value, new_edge_index))
                elif key == 'edge_attr':
                    value = torch.vstack((value, new_edge_attr))
                elif key == 'concealer_edge_mask':
                    value = torch.vstack((value, torch.zeros(new_edge_index.shape[1], 1).bool()))
                elif key == 'rare_edge_mask':
                    value = torch.vstack((value, torch.zeros(new_edge_index.shape[1], 1).bool()))
                else:
                    raise NotImplementedError(key)
                keys.append(key)
                values.append(value)
        for key, value in zip(keys, values):
            store[key] = value
    return graph.sort()
