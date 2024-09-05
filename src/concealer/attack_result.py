from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from blosc_compress import blosc_pkl_dump, blosc_pkl_load
from src.concealer.analysis.test_metrics import TestMetrics


def attack_result(
    experiment_dir: Path,
    pids_name=None,
    num_lowest: float = 0.1,
    device: str = 'cuda',
    mode: str = 'simple',
):
    assert isinstance(num_lowest, float)
    txt_path = experiment_dir.joinpath(f'top_{int(num_lowest * 100)}p_attack_results.txt')
    if mode == 'simple':
        device = 'cpu'
    print(experiment_dir.parts[-1])
    if 'atlas' in str(experiment_dir):
        dataset_name = 'atlas'
        batch_size = 32
    elif 'supply' in str(experiment_dir):
        dataset_name = 'supply'
        batch_size = 64
    else:
        raise KeyError(str(experiment_dir))
    evaluator = TestMetrics(dataset_name=dataset_name, device=device, batch_size=batch_size, num_evasion=True)
    if pids_name is None:
        pids_names = list(evaluator.threshold.keys())
        decrease_rate = {pids_name: 0. for pids_name in evaluator.threshold}
        evasion_rate = {pids_name: 0. for pids_name in evaluator.threshold}
        total_perturbation = {pids_name: 0. for pids_name in evaluator.threshold}
        num_graphs = {pids_name: 0 for pids_name in evaluator.threshold}
    else:
        pids_names = [pids_name]
        decrease_rate = {pids_name: 0.}
        evasion_rate = {pids_name: 0.}
        total_perturbation = {pids_name: 0.}
        num_graphs = {pids_name: 0}
    for graph_dir in tqdm(natsorted(experiment_dir.iterdir(), key=lambda x: x.stem)):
        if not graph_dir.is_dir():
            continue
        if mode == 'complex':
            examples, _ = blosc_pkl_load(graph_dir.joinpath(f'{graph_dir.stem}.blosc'))
            anomaly_scores, num_evasion = evaluator(examples, non_edge=False)
        elif mode == 'simple':
            examples, anomaly_scores = blosc_pkl_load(graph_dir.joinpath(f'{graph_dir.stem}.blosc'))
        else:
            raise KeyError(mode)
        # update anomaly_score.csv
        nodes_modified = [0]
        edges_modified = [0]
        original_graph = examples[0]
        adversarial_graphs = examples[1:]
        for data in adversarial_graphs:
            nodes_modified.append(abs(data.num_nodes - original_graph.num_nodes))
            if hasattr(data, 'num_rare_edge_modified'):
                edges_modified.append(data.num_edges - original_graph.num_edges + data.num_rare_edge_modified)
            else:
                edges_modified.append(abs(data.num_edges - original_graph.num_edges))
        csv_path = graph_dir.joinpath('anomaly_score.csv')
        anomaly_scores['nodes_modified'] = nodes_modified
        anomaly_scores['edges_modified'] = edges_modified
        df = pd.DataFrame(anomaly_scores)
        df = df.round(6)
        df = df.round({'nodes_modified': 0})
        df = df.round({'edges_modified': 0})
        df.to_csv(str(csv_path), index=False)
        # update num_evasion.csv
        if mode == 'complex':
            csv_path = graph_dir.joinpath('num_evasion.csv')
            df = pd.DataFrame(num_evasion, index=[0])
            df.to_csv(str(csv_path), index=False)
        # update *.blosc
        if mode == 'complex':
            blosc_pkl_dump([examples, anomaly_scores], graph_dir.joinpath(f'{graph_dir.stem}.blosc'))
        # perturbation size
        nodes_modified = np.array(nodes_modified, dtype=np.int32)
        edges_modified = np.array(edges_modified, dtype=np.int32)
        adversarial_perturbations = (np.abs(nodes_modified[1:]) + np.abs(edges_modified[1:])) / (original_graph.num_nodes +
                                                                                                 original_graph.num_edges)
        # attack results
        for pids_name in pids_names:
            scores = anomaly_scores[pids_name]
            original_score = np.array(scores[0])
            adversarial_scores = np.array(scores[1:])
            threshold = np.array(evaluator.threshold[pids_name])
            if original_score < threshold:
                continue
            num_graphs[pids_name] += 1
            num_lowest_ = int(np.ceil(len(adversarial_scores) * num_lowest))
            if num_lowest_ < 0:
                num_lowest_ = 1
            if num_lowest_ > len(adversarial_scores):
                num_lowest_ = len(adversarial_scores)
            if num_lowest_ == 1:
                target_index = np.lexsort((adversarial_perturbations, adversarial_scores))[:num_lowest_]
            else:
                target_index = np.argsort(adversarial_scores)[:num_lowest_]
            adversarial_scores = adversarial_scores[target_index]
            decrease_rate[pids_name] += (sum((adversarial_scores - original_score) / original_score) / num_lowest_)
            evasion_rate[pids_name] += (sum(adversarial_scores <= threshold) / num_lowest_)
            total_perturbation[pids_name] += (sum(adversarial_perturbations[target_index]) / num_lowest_)

    # attack_result.txt
    for pids_name in pids_names:
        if num_graphs[pids_name] == 0:
            num_graphs[pids_name] = 0.5
        decrease_rate[pids_name] /= num_graphs[pids_name]
        evasion_rate[pids_name] /= num_graphs[pids_name]
        total_perturbation[pids_name] /= num_graphs[pids_name]
    with txt_path.open('w') as fw:
        fw.write(f'evasion_rate: {evasion_rate}\n')
        fw.write(f'decrease_rate: {decrease_rate}\n')
        fw.write(f'total_perturbation: {total_perturbation}\n')
    return


if __name__ == '__main__':
    experiment_dir = Path('adversarial_graphs/supply/fga-GATv2-concealer')
    attack_result(experiment_dir, num_lowest=0.1, mode='simple')
