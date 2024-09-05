import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from deap import base, creator, tools
from natsort import natsorted
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm

from blosc_compress import blosc_pkl_dump, blosc_pkl_load
from src.pids.fga.utils import AutoencoderEvaluator
from src.pids.flash.utils import FlashEvaluator

warnings.filterwarnings('ignore')
MAX_CONCEALER_SIZE = 0.1


def custom_largest_weakly_cc(data: Data, connection='weak'):
    assert data.edge_index is not None

    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    _, component = sp.csgraph.connected_components(adj, connection=connection)

    _, count = np.unique(component, return_counts=True)
    subset_np = np.in1d(component, count.argsort()[-1:])
    subset = torch.from_numpy(subset_np)
    subset = subset.to(data.edge_index.device, torch.bool)

    return data.subgraph(subset), subset


def pruner(dataset_name, pids_name, gnn_type=None):
    # args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # anomaly score evaluator
    if pids_name == 'flash':
        evaluator = FlashEvaluator(dataset_name, device=device, batch_size=32)
    elif pids_name == 'fga':
        assert gnn_type in ['GATv2', 'GCN']
        evaluator = AutoencoderEvaluator(dataset_name, gnn_type, device=device, batch_size=32)
    else:
        raise NotImplementedError(pids_name)
    # data: pyg homograph
    if pids_name == 'flash':
        saved_benign_path = Path(f'dataset/{dataset_name}/pids/{pids_name}/raw_benign_graphs.pt')
    elif pids_name == 'fga':
        saved_benign_path = Path(f'dataset/{dataset_name}/pids/{pids_name}/benign_graphs.pt')
    if saved_benign_path.exists():
        benign_graphs = torch.load(saved_benign_path)
    else:
        benign_graphs = []
        benign_dir = Path(f'dataset/{dataset_name}/processed/benign')
        file_iter = natsorted(benign_dir.glob('*.blosc'), key=lambda x: x.stem)
        pbar = tqdm(file_iter, desc='processing benign graphs', leave=False)
        for graph_path in pbar:
            pbar.set_description(f'convert data format: graph_{graph_path.stem}')
            nx_graph = blosc_pkl_load(graph_path)
            if pids_name == 'flash':
                data = evaluator.parser.processed2predictor(nx_graph)
            elif pids_name == 'fga':
                data = evaluator.parser.processed2fga(nx_graph)
            data.graph_id = graph_path.stem
            benign_graphs.append(data)
        if not saved_benign_path.parent.exists():
            saved_benign_path.parent.mkdir(parents=True)
        torch.save(benign_graphs, saved_benign_path)
    # trainset (concealer masks) generation
    if pids_name == 'flash':
        saved_dir = Path(f'dataset/{dataset_name}/graph_pruning/{pids_name}')
    elif pids_name == 'fga':
        saved_dir = Path(f'dataset/{dataset_name}/graph_pruning/{pids_name}/{gnn_type}')
    if not saved_dir.exists():
        saved_dir.mkdir(parents=True)
    epoch_iter = tqdm(benign_graphs, file=sys.stdout)
    for data in epoch_iter:
        epoch_iter.set_description(f'pruning: graph_{data.graph_id}')
        if saved_dir.joinpath(f'{data.graph_id}.blosc').exists():
            continue
        else:
            concealer_masks = graph_pruning(data, evaluator)
            # save concealer masks (i.e., binary strings)
            blosc_pkl_dump(concealer_masks, saved_dir.joinpath(f'{data.graph_id}.blosc'))
    return


def eval_pids(pop, g: Data, evaluator):
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
        if context_graph.num_nodes and context_graph.num_edges > g.num_edges / 2:
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


def largest_connected_subgraph(g: Data, pop):
    r'''update individuals
    '''
    pop = np.array(pop, dtype=np.bool_)
    if pop.ndim == 1:
        pop = np.expand_dims(pop, axis=0)
    context_graphs = []
    concealer_masks = []
    for ind in pop:
        concealer_mask = ind.copy()
        if concealer_mask.max() == 0:
            context_graph = g.clone()
            concealer_mask = np.array(concealer_mask, dtype=np.bool_)
        else:
            context_mask = np.zeros(g.num_nodes, dtype=np.bool_)
            concealer_mask = np.array(concealer_mask, dtype=np.bool_)
            remained_nids = (~concealer_mask).nonzero()[0]
            assert (remained_nids == np.sort(remained_nids)).all()
            attacked_graph = g.subgraph(torch.from_numpy(remained_nids))
            if attacked_graph.num_edges:
                _, context_flags = custom_largest_weakly_cc(attacked_graph)
                context_flags = context_flags.numpy()
                if sum(context_flags) < attacked_graph.num_nodes:
                    context_nids = remained_nids[context_flags]
                    context_mask[context_nids] = True
                    concealer_mask = ~context_mask
                else:
                    context_mask = ~concealer_mask
            else:
                concealer_mask = np.ones_like(concealer_mask, dtype=np.bool_)
            context_graph = g.subgraph(torch.from_numpy(context_mask))
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
                smart_hof_idx = np.argmax(hof_values) if hof_values else 0
                smart_removed_nodes = round(halloffame.keys[smart_hof_idx].values[0] * len(ind))
                smart_anomaly_score = halloffame.keys[smart_hof_idx].values[1]
                tqdm.write(f'{gen}|{ngen} smart_ind: num_removed_nodes{smart_removed_nodes} anomaly_score={smart_anomaly_score:.8f}')
        # Select the next generation pop
        pop[:] = toolbox.select(pop + offspring, mu)

    return pop, original_anomaly_score


def graph_pruning(g: Data, evaluator):
    # Args
    lambda_ = 100  # the number of children to produce at each generation
    ind_size = g.num_nodes  # the numbers of nodes (graph size)
    # Creating Types
    creator.create('Fitness', base.Fitness, weights=(-1.0, 1.0))  # for fitness function
    creator.create('Individual', list, fitness=creator.Fitness)
    # Initialization
    toolbox = base.Toolbox()
    # Initialize individual: 99% as False, 1% as True. False represents context nodes, and True represents concealer nodes.
    toolbox.register('attr_bool', lambda: False if random.uniform(0, 1) < 0.99 else True)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, n=ind_size)
    toolbox.register('pop', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', eval_pids, g=g, evaluator=evaluator)
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
    dataset_name = 'atlas'  # no SupplyChain
    pids_name = 'flash'
    pruner(dataset_name, pids_name)
