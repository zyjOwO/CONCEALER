import math
from pathlib import Path

import torch
from natsort import natsorted
from torch.utils.data import Dataset, random_split
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from tqdm import tqdm

from blosc_compress import blosc_pkl_dump, blosc_pkl_load
from dataset.supply.parser import Parser
from src.pids.fga.utils import AutoencoderEvaluator
from src.pids.flash.utils import FlashEvaluator
from src.pids.sgat.utils import SGATEvaluator

from src.concealer.data.utils import (
    add_context_reverse_edges,
    apply_noise,
    compute_stats,
)

MAX_CONCEALER_SIZE = 0.1


class SupplyDataModule(LightningDataset):

    def __init__(self, data_dir, config, num_workers=2):
        self.config = config
        self.batch_size = config['train']['batch_size']
        self.num_workers = num_workers if num_workers < self.batch_size else self.batch_size

        self.pruning_data_dir = data_dir
        temp_dir = data_dir.parent.parent if 'fga' in str(self.pruning_data_dir) else data_dir.parent
        self.processed_data_dir = Path(str(temp_dir).replace('graph_pruning', 'processed'))
        self.concealer_data_dir = Path(str(data_dir).replace('graph_pruning', 'concealer'))

        self.parser = Parser('supply')
        self.node_dim = len(self.parser.valid_ntypes)
        self.edge_dim = len(self.parser.valid_etypes) + 1  # raw_edge_dim + 1 (non_edge)

        self.stats = None

        if 'fga' in str(self.pruning_data_dir):
            if 'GCN' in str(self.pruning_data_dir):
                self.threshold = AutoencoderEvaluator('supply', 'GCN').threshold
            elif 'GATv2' in str(self.pruning_data_dir):
                self.threshold = AutoencoderEvaluator('supply', 'GATv2').threshold
        elif 'sgat' in str(self.pruning_data_dir):
            self.threshold = SGATEvaluator('supply').threshold
        elif 'flash' in str(self.pruning_data_dir):
            self.threshold = FlashEvaluator('supply').threshold
        else:
            raise NotImplementedError

        self.prepare_data()
        self.setup('fit')
        self.setup('test')

        super().__init__(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def read_processed_data(self, graph_dir):
        # pyg homograph -> one-hot encoding -> valid concealer_masks -> *.pt file
        file_iter = natsorted(graph_dir.glob('*.blosc'), key=lambda x: int(x.stem))
        tqdm_iter = tqdm(file_iter, desc=f'reading {graph_dir.name} graphs with concealers')
        for graph_path in tqdm_iter:
            # nx_digraph -> pyg homograph with one-encoded features
            nx_graph = blosc_pkl_load(graph_path)
            graph = self.parser.processed2concealer(nx_graph, non_edge=True)  # non-edge
            graph.graph_name = int(graph_path.stem)

            # get concealer_masks
            if graph_dir.name == 'benign':
                graph.concealer_masks = []
                concealer_masks = blosc_pkl_load(self.concealer_data_dir.joinpath(f'{graph_path.stem}.blosc'))
                # filter out false-positive graphs
                original_score = concealer_masks[0].fitness.values[1]
                if original_score > self.threshold:
                    continue
                # save only the masks where the anomaly value is greater than the original anomaly value
                max_concealer_size = math.ceil(graph.num_nodes * MAX_CONCEALER_SIZE)
                for mask in concealer_masks[1:]:
                    if mask.fitness.values[1] >= self.threshold:
                        if sum(mask) <= max_concealer_size:
                            graph.concealer_masks.append(mask)
                if not graph.concealer_masks:
                    continue
                graph.concealer_masks = torch.tensor(graph.concealer_masks, dtype=torch.bool)
                assert torch.unique(graph.concealer_masks, dim=0).shape[0] == graph.concealer_masks.shape[0]
            yield graph

    def prepare_data(self):
        if self.concealer_data_dir.exists():
            return
        for glabel_dir in self.processed_data_dir.iterdir():
            if not glabel_dir.is_dir() or glabel_dir.name not in ['benign', 'anomaly']:
                continue
            saved_dir = self.concealer_data_dir.joinpath(glabel_dir.name)
            saved_dir.mkdir(parents=True)
            for graph in self.read_processed_data(glabel_dir):
                save_path = saved_dir.joinpath(f'{graph.graph_name}.pt')
                torch.save(graph, save_path)
        return

    def setup(self, stage: str):
        if self.stats is not None:
            return
        benign_dir = self.concealer_data_dir.joinpath('benign')
        anomaly_dir = self.concealer_data_dir.joinpath('anomaly')

        if self.stats is None:
            stats_path = benign_dir.joinpath('stats.blosc')
            if stats_path.exists():
                self.stats = blosc_pkl_load(stats_path)
            else:
                graphs = [torch.load(graph_dir) for graph_dir in benign_dir.glob('*.pt')]
                self.stats = compute_stats(graphs, self.parser.valid_ntypes, self.parser.valid_etypes)
                blosc_pkl_dump(stats_path)
                del graphs

        benign_dataset = SupplyDataset(benign_dir, self.config, 'benign')
        anomaly_dataset = SupplyDataset(anomaly_dir, self.config, 'anomaly')

        generator = torch.Generator().manual_seed(2024)
        self.train_dataset, self.val_dataset = random_split(benign_dataset, [0.8, 0.2], generator)
        self.test_dataset = anomaly_dataset


class SupplyDataset(Dataset):

    def __init__(self, root: Path, config: dict, glabel: str):
        self.root = root
        self.graph_paths = list(sorted(self.root.glob('*.pt'), key=lambda x: int(x.stem)))
        self.glabel = glabel
        # config
        self.model_config = config['model']
        self.diffusion_config = config['diffusion']
        # bidirectional edges
        self.bidirected = add_context_reverse_edges if self.model_config['bidirected'] else None
        # positional encoding
        self.pos_enc = None
        if self.model_config['pos_enc'] == 'laplacian':
            self.pos_enc = AddLaplacianEigenvectorPE(
                k=self.model_config['pos_enc_dim'],
                attr_name=None,
                is_undirected=self.model_config['bidirected'],
            )
        elif self.model_config['pos_enc'] == 'random_walk':
            self.pos_enc = AddRandomWalkPE(
                walk_length=self.model_config['pos_enc_dim'],
                attr_name=None,
            )
        super().__init__()

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, index):
        graph = torch.load(self.graph_paths[index])
        graph.x = graph.x.to(torch.float32)
        graph.edge_index = graph.edge_index.to(torch.int64)
        graph.edge_attr = graph.edge_attr.to(torch.float32)

        if self.glabel == 'benign':
            mask_index = torch.randint(0, graph.concealer_masks.shape[0], (1,))[0]
            graph.concealer_mask = graph.concealer_masks[mask_index]
            del graph.concealer_masks

            noisy_graph, clean_graph = apply_noise(graph, **self.diffusion_config)

            if self.bidirected is not None:
                noisy_graph = self.bidirected(noisy_graph)
                clean_graph = self.bidirected(clean_graph)

            if self.pos_enc is not None:
                noisy_graph = self.pos_enc(noisy_graph)

            assert torch.equal(noisy_graph.edge_index, clean_graph.edge_index)
            assert torch.equal(noisy_graph.concealer_edge_mask, clean_graph.concealer_edge_mask)
            assert torch.equal(noisy_graph.concealer_node_mask, clean_graph.concealer_node_mask)
            clean_graph.graph_name = graph.graph_name
            return noisy_graph, clean_graph
        elif self.glabel == 'anomaly':
            return graph
        else:
            raise ValueError(self.glabel)
