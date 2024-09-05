from copy import deepcopy

import torch

from src.pids.fga.utils import AutoencoderEvaluator
from src.pids.flash.utils import FlashEvaluator
from src.pids.sgat.utils import SGATEvaluator


class TestMetrics():

    def __init__(self, dataset_name, device='cpu', batch_size=4, num_evasion=False, show=False):
        self.dataset_name = dataset_name
        self.device = device
        # PIDS
        self.sgat_eval = SGATEvaluator(
            dataset_name,
            device=device,
            batch_size=batch_size,
        )
        self.fga_gcn_eval = AutoencoderEvaluator(
            dataset_name,
            gnn_type='GCN',
            device=device,
            batch_size=batch_size,
        )
        self.fga_gat_eval = AutoencoderEvaluator(
            dataset_name,
            gnn_type='GATv2',
            device=device,
            batch_size=batch_size,
        )
        self.flash_eval = FlashEvaluator(
            dataset_name,
            device=device,
            batch_size=batch_size,
        )
        # anomaly score threshold
        self.threshold = {}
        self.threshold['sgat'] = self.sgat_eval.threshold
        self.threshold['fga_gcn'] = self.fga_gcn_eval.threshold
        self.threshold['fga_gat'] = self.fga_gat_eval.threshold
        self.threshold['flash'] = self.flash_eval.threshold
        # show
        self.num_evasion = num_evasion
        self.show = show

    def __call__(self, examples: list, non_edge=True):
        examples = deepcopy(examples)
        if isinstance(examples, list) is False:
            examples = [examples]

        # anomaly scores
        anomaly_scores = {}
        # SGAT
        parsed_examples = []
        for example in examples:
            parsed_example = self.sgat_eval.parser.concealer2sgat(example.clone(), non_edge=non_edge)[0]
            parsed_examples.append(parsed_example)
        anomaly_scores['sgat'] = self.sgat_eval(parsed_examples)
        # FGA-GCN FGA-GATv2
        parsed_examples = []
        for example in examples:
            parsed_examples.append(self.fga_gcn_eval.parser.concealer2fga(example.clone(), non_edge=non_edge))
        anomaly_scores['fga_gcn'] = self.fga_gcn_eval(parsed_examples)
        anomaly_scores['fga_gat'] = self.fga_gat_eval(parsed_examples)
        # FLASH
        anomaly_scores['flash'] = self.flash_eval(examples)

        # num_evasion
        if self.num_evasion:
            num_evasion = {}
            for k, v in anomaly_scores.items():
                if len(v) > 1:
                    num_evasion[k] = str(sum(torch.tensor(v[1:]) < self.threshold[k]).item())
                else:
                    num_evasion[k] = str(0)
                if torch.tensor(v[0]) < self.threshold[k]:  # label * represents false-positive graph
                    num_evasion[k] = f'{num_evasion[k]}*'
            if self.show:
                print(f'num_evasion: {num_evasion}')
            return anomaly_scores, num_evasion
        else:
            return anomaly_scores
