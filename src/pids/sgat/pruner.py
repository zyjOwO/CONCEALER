import math
import random
import sys
import warnings
from pathlib import Path

import dgl
import numpy as np
import torch
from deap import base, creator, tools
from dgl import DGLHeteroGraph
from natsort import natsorted
from tqdm import tqdm

from blosc_compress import blosc_pkl_dump, blosc_pkl_load
from src.pids.sgat.utils import SGATEvaluator

warnings.filterwarnings('ignore')
MAX_CONCEALER_SIZE = 0.1


def weakly_connected_components(g: DGLHeteroGraph):
    g = dgl.to_bidirected(g)
    seen = set()
    for source in g.nodes():
        if source.tolist() not in seen:
            c = torch.cat(dgl.bfs_nodes_generator(g, source)).tolist()
            c = sorted(c)
            seen.update(c)
            yield c


def pruner(dataset_name, pids_name='sgat'):
    # args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bidirection = True
    # anomaly score evaluator
    evaluator = SGATEvaluator(dataset_name, bidirection=bidirection, device=device, batch_size=32)
    # data: dgl heterograph
    dgl_benign_path = Path(f'dataset/{dataset_name}/pids/{pids_name}/benign_graphs.blosc')
    if dgl_benign_path.exists():
        benign_graphs, benign_graph_names, benign_nid_maps = blosc_pkl_load(dgl_benign_path)
    else:
        benign_graphs = []
        benign_nid_maps = []
        benign_graph_names = []
        benign_dir = Path(f'dataset/{dataset_name}/processed/benign')
        file_iter = natsorted(benign_dir.glob('*.blosc'), key=lambda x: x.stem)
        pbar = tqdm(file_iter, desc='reading benign graphs')
        for graph_path in pbar:
            nx_graph = blosc_pkl_load(graph_path)
            graph, nid_map = evaluator.parser.processed2sgat(nx_graph)
            benign_graphs.append(graph)
            benign_nid_maps.append(nid_map)
            benign_graph_names.append(graph_path.stem)
        blosc_pkl_dump((benign_graphs, benign_graph_names, benign_nid_maps), dgl_benign_path)
    # trainset (concealer masks) generation
    saved_dir = Path(f'dataset/{dataset_name}/graph_pruning/{pids_name}')
    if not saved_dir.exists():
        saved_dir.mkdir(parents=True)
    benign_graphs = dict(zip(benign_graph_names, benign_graphs))
    benign_nid_maps = dict(zip(benign_graph_names, benign_nid_maps))
    pbar = tqdm(benign_graphs, file=sys.stdout)
    for graph_name in pbar:
        pbar.set_description(f'pruning: graph_{graph_name}')
        if saved_dir.joinpath(f'{graph_name}.blosc').exists():
            continue
        else:  # graph pruning
            graph = benign_graphs[graph_name]
            nid_map = benign_nid_maps[graph_name]
            concealer_masks = graph_pruning(graph, evaluator)
            concealer_masks = evaluator.parser.concealer_nid_mapping(graph, concealer_masks, nid_map)
            # save concealer masks (i.e., binary strings)
            blosc_pkl_dump(concealer_masks, saved_dir.joinpath(f'{graph_name}.blosc'))
    return


def eval_sgat(pop, g: DGLHeteroGraph, evaluator):
    weights = []  # concealer size
    values = []  # anomaly score
    # update individuals
    context_graphs, concealer_masks = largest_connected_subgraph(g, pop)
    for idx, ind in enumerate(pop):
        ind.clear()
        ind.extend(concealer_masks[idx])
    # filter out small graphs
    is_valid = np.zeros(len(pop), dtype=np.bool_)
    for valid_idx, context_graph in enumerate(context_graphs):
        if context_graph.num_nodes() and context_graph.num_edges() > g.num_edges() / 2:
            is_valid[valid_idx] = True
    # evaluation
    concealer_masks = np.array(concealer_masks, dtype=np.bool_)
    valid_context_graphs = [context_graphs[idx] for idx in is_valid.nonzero()[0]]
    valid_concealer_masks = concealer_masks[is_valid.nonzero()[0]]
    if len(valid_context_graphs) > 0:
        anomaly_scores = evaluator(valid_context_graphs)
        values.extend([anomaly_scores.pop(0) if x else 0. for x in is_valid])
        weight = [sum(ind) / len(ind) for ind in valid_concealer_masks]
        weights.extend([weight.pop(0) if x else 10000. for x in is_valid])
    else:
        values.extend([0.] * len(is_valid))
        weights.extend([10000.] * len(is_valid))
    return weights, values


def largest_connected_subgraph(g: DGLHeteroGraph, pop):
    r'''update individuals
    '''
    pop = np.array(pop, dtype=np.bool_)
    if pop.ndim == 1:
        pop = np.expand_dims(pop, axis=0)
    context_graphs = []
    concealer_masks = []
    for ind in pop:
        concealer_mask = ind.copy()
        homo_graph = dgl.unbatch(dgl.to_homogeneous(g))[0]  # heterograph -> homograph
        if concealer_mask.max() == 0:
            context_graph = g.clone()
            concealer_mask = np.array(concealer_mask, dtype=np.bool_)
        else:
            removed_nids = torch.as_tensor(concealer_mask.nonzero()[0])
            attacked_graph = dgl.remove_nodes(homo_graph, nids=removed_nids, store_ids=True)
            context_mask = np.zeros(homo_graph.num_nodes(), dtype=np.bool_)
            if attacked_graph.num_edges():
                context_nids = max(list(weakly_connected_components(attacked_graph)), key=len)
                context_nids = attacked_graph.ndata[dgl.NID][context_nids]
                context_mask[context_nids] = True
                concealer_mask = ~context_mask
            else:
                concealer_mask = np.ones_like(concealer_mask, dtype=np.bool_)
            context_graph = dgl.unbatch(g.clone())[0]
            ntype_count = [g.num_nodes(ntype) for ntype in g.ntypes]
            ntype_offset = np.insert(np.cumsum(ntype_count), 0, 0)
            for idx, ntype in enumerate(g.ntypes):
                nids = np.where(concealer_mask[ntype_offset[idx]:ntype_offset[idx + 1]])[0]
                if nids.shape[0] > 0:
                    context_graph.remove_nodes(nids=torch.as_tensor(nids), ntype=ntype)
        context_graphs.append(context_graph)
        concealer_masks.append(concealer_mask.tolist())

    if len(concealer_masks) == 1:
        return context_graphs[0], concealer_masks[0]
    else:
        return context_graphs, concealer_masks


def mutate(ind):
    ''' one-bit flip '''
    idx = random.randint(0, len(ind) - 1)
    ind[idx] = not ind[idx]
    return ind


def varOr(pop, toolbox, lambda_, cxpb, mutpb):
    ''' apply crossover, mutation, or reproduction function '''
    assert (cxpb + mutpb) <= 1.0
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        # Apply crossover
        if op_choice < cxpb:
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(pop, 2)]
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        # Apply mutation
        elif op_choice < cxpb + mutpb:
            ind = toolbox.clone(random.choice(pop))
            ind = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        # Apply reproduction
        else:
            offspring.append(random.choice(pop))
    # Concealer size limitation
    for child in offspring:
        child_size = sum(child)
        max_child_size = math.ceil(MAX_CONCEALER_SIZE * len(child))
        if child_size > max_child_size:
            child_np = np.array(child, dtype=np.bool_)
            indices = child_np.nonzero()[0]
            num_removed = child_size - max_child_size
            indices = np.random.choice(indices, size=num_removed, replace=False)
            child_np[indices] = False
            child.clear()
            child.extend(child_np.tolist())
    return offspring


def eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.5, mutpb=0.5, ngen=100, halloffame=None):
    ''' This is the :math:`(\mu + \lambda)` evolutionary algorithm.
        pop: A list of individuals.
        toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
        mu:  The number of individuals to select for the next generation.
        lambda_: The number of children to produce at each generation.
        cxpb: The probability that an offspring is produced by crossover.
        mutpb: The probability that an offspring is produced by mutation.
        ngen: The number of generation.
        halloffame: A :class:`~deap.tools.HallOfFame` object that will contain the best individuals, optional.
        returns: The final pop
        returns: A class:`~deap.tools.Logbook` with the statistics of the evolution.
    '''
    # 1. Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.evaluate(invalid_ind)
    for idx, ind in enumerate(invalid_ind):
        ind.fitness.values = (fitnesses[0][idx], fitnesses[1][idx])
    original_anomaly_score = fitnesses[1][0]
    if halloffame is not None:
        halloffame.update(pop)
        hof_values = [sum(key.wvalues) for key in halloffame.keys[:-1]]
        smart_hof_idx = np.argmax(hof_values) if hof_values else 0
        smart_removed_nodes = round(halloffame.keys[smart_hof_idx].values[0] * len(ind))
        smart_anomaly_score = halloffame.keys[smart_hof_idx].values[1]
        tqdm.write(f'original_num_nodes={len(ind)}, original_anomaly_score={original_anomaly_score:.8f}')
        tqdm.write(f'0|{ngen} smart_ind: num_removed_nodes={smart_removed_nodes} anomaly_score={smart_anomaly_score:.8f}')

    # 2. Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the pop
        offspring = varOr(pop, toolbox, lambda_, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.evaluate(invalid_ind)
        for idx, ind in enumerate(invalid_ind):
            ind.fitness.values = (fitnesses[0][idx], fitnesses[1][idx])
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            if gen % 10 == 0:
                hof_values = [sum(key.wvalues) for key in halloffame.keys[:-1]]
                assert halloffame.keys[-1].values[0] == 0.
                smart_hof_idx = np.argmax(hof_values) if hof_values else 0
                smart_removed_nodes = round(halloffame.keys[smart_hof_idx].values[0] * len(ind))
                smart_anomaly_score = halloffame.keys[smart_hof_idx].values[1]
                tqdm.write(f'{gen}|{ngen} smart_ind: num_removed_nodes={smart_removed_nodes} anomaly_score={smart_anomaly_score:.8f}')
        # Select the next generation pop
        pop[:] = toolbox.select(pop + offspring, mu)

    return pop, original_anomaly_score


def graph_pruning(g: DGLHeteroGraph, evaluator):
    # Args
    lambda_ = 100  # the number of children to produce at each generation
    ind_size = g.num_nodes()  # the numbers of nodes (graph size)
    # Creating Types
    creator.create('Fitness', base.Fitness, weights=(-1.0, 1.0))  # for fitness function
    creator.create('Individual', list, fitness=creator.Fitness)
    # Initialization
    toolbox = base.Toolbox()
    # Initialize individual: 99% as False, 1% as True. False represents context nodes, and True represents concealer nodes.
    toolbox.register('attr_bool', lambda: False if random.uniform(0, 1) < 0.99 else True)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, n=ind_size)
    toolbox.register('pop', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', eval_sgat, g=g, evaluator=evaluator)
    # Operators
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', mutate)
    toolbox.register('select', tools.selNSGA2)

    # Evolution
    pop = toolbox.pop(n=lambda_)
    pop[0] = creator.Individual([False] * len(pop[0]))
    hof = tools.ParetoFront()
    pop, original_anomaly_score = eaMuPlusLambda(pop, toolbox, halloffame=hof)

    # Remove solutions with anomaly_score smaller than original_anomaly_score
    for index in range(len(hof.items)):
        if hof.items[index].fitness.values[1] < original_anomaly_score:
            hof.remove(index)
    return hof.items


if __name__ == '__main__':
    dataset_name = 'supply'
    pids_name = 'sgat'
    pruner(dataset_name, pids_name)
