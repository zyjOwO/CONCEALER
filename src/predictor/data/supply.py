import math
from pathlib import Path

import torch
from natsort import natsorted
from torch.utils.data import Dataset, random_split
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
    ToUndirected,
)
from tqdm import tqdm

from blosc_compress import blosc_pkl_dump, blosc_pkl_load
from dataset.supply.parser import Parser
from src.pids.fga.utils import AutoencoderEvaluator
from src.pids.sgat.utils import SGATEvaluator

from .utils import compute_stats, get_binding_sites, get_concealer_size

MAX_CONCEALER_SIZE = 0.1


class SupplyDataModule(LightningDataset):

    def __init__(
        self,
        data_dir: Path,
        task: str,
        batch_size=4,
        num_workers=2,
        bidirection=True,
        pos_enc=True,
        **kwargs,
    ):
        self.task = task  # binding_site or concealer_size
        self.dataset_name = 'supply'
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers < batch_size else batch_size
        self.bidirection = bidirection
        self.pos_enc = pos_enc
        self.pos_enc_dim = 0
        if pos_enc:
            self.pos_enc_dim = kwargs['pos_enc_dim']
            assert self.pos_enc_dim > 0

        # paths
        self.pruning_data_dir = data_dir
        self.processed_data_dir = Path(*data_dir.parts[:data_dir.parts.index('graph_pruning')]).joinpath('processed')
        self.predictor_data_dir = Path(str(data_dir).replace('graph_pruning', 'predictor'))

        # PIDS and anomaly threshold
        self.gnn_type = None
        if 'fga' in data_dir.parts:
            self.pids_name = 'fga'
            self.gnn_type = data_dir.parts[-1]
            self.threshold = AutoencoderEvaluator(self.dataset_name, self.gnn_type).threshold
        elif 'flash' in data_dir.parts:
            raise NotImplementedError('flash is deprecated for supply dataset')
        elif 'sgat' in data_dir.parts:
            self.pids_name = 'sgat'
            self.threshold = SGATEvaluator(self.dataset_name).threshold
        else:
            raise NotImplementedError('cannot find valid pids name in the data_dir')

        # parser
        self.parser = Parser(self.dataset_name)
        self.node_dim = len(self.parser.valid_ntypes)
        self.edge_dim = len(self.parser.valid_etypes)

        # stats
        self.stats = None

        self.prepare_data()
        self.setup()
        super().__init__(self.train_dataset, self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def read_data_with_concealer_mask(self, graph_dir: Path):
        file_iter = natsorted(graph_dir.glob('*.blosc'), key=lambda x: x.stem)
        tqdm_iter = tqdm(file_iter, desc='reading benign graph with concealer_masks')
        for graph_path in tqdm_iter:
            # nx_graph
            nx_graph = blosc_pkl_load(graph_path)
            graph = self.parser.processed2predictor(nx_graph)
            graph.graph_name = graph_path.stem
            graph.concealer_masks = []

            # concealer_masks
            concealer_masks = blosc_pkl_load(self.pruning_data_dir.joinpath(f'{graph_path.stem}.blosc'))

            # filter out false-positive graphs
            threshold = self.threshold / graph.num_nodes if self.threshold > 1. else self.threshold
            original_score = concealer_masks[0].fitness.values[1]
            if original_score > threshold:
                continue

            # save only the masks where the anomaly value is greater than the original anomaly value
            max_concealer_size = math.ceil(graph.num_nodes * MAX_CONCEALER_SIZE)
            x, y = [], []
            for mask in concealer_masks[1:]:
                if mask.fitness.values[1] >= original_score:
                    if sum(mask) <= max_concealer_size:
                        graph.concealer_masks.append(mask)
                        x.append(sum(mask))
                        y.append(mask.fitness.values[1])
            if not graph.concealer_masks:
                continue
            graph.concealer_masks = torch.as_tensor(graph.concealer_masks)
            yield graph

    def prepare_data(self):
        if self.predictor_data_dir.exists():
            return
        for glabel_dir in self.processed_data_dir.iterdir():
            if not glabel_dir.is_dir() or glabel_dir.name != 'benign':
                continue
            saved_dir = self.predictor_data_dir.joinpath(glabel_dir.name)
            if not saved_dir.exists():
                saved_dir.mkdir(parents=True)
            for graph in self.read_data_with_concealer_mask(glabel_dir):
                save_path = saved_dir.joinpath(f'{graph.graph_name}.pt')
                torch.save(graph, save_path)
        return

    def setup(self, stage=None):
        if self.stats is not None:
            return
        benign_dir = self.predictor_data_dir.joinpath('benign')
        stats_path = benign_dir.joinpath('stats.blosc')
        if stats_path.exists():
            self.stats = blosc_pkl_load(stats_path)
        else:
            graphs = (torch.load(dir) for dir in benign_dir.glob('*.pt'))
            self.stats = compute_stats(graphs)
            blosc_pkl_dump(self.stats, stats_path)

        dataset = SupplyDataset(benign_dir, self.task, self.bidirection, self.pos_enc, self.pos_enc_dim)
        generator = torch.Generator().manual_seed(2024)
        self.train_dataset, self.val_dataset = random_split(dataset, [0.8, 0.2], generator)


class SupplyDataset(Dataset):

    def __init__(
        self,
        root: Path,
        task: str,
        bidirection: bool,
        pos_enc: bool,
        pos_enc_dim: int,
    ):
        super().__init__()
        self.root = root
        self.task = task
        self.bidirection = ToUndirected() if bidirection else None
        self.pos_enc = None
        if pos_enc == 'laplacian':
            self.pos_enc = AddLaplacianEigenvectorPE(k=pos_enc_dim, attr_name=None, is_undirected=bidirection)
        elif pos_enc == 'random_walk':
            self.pos_enc = AddRandomWalkPE(walk_length=pos_enc_dim, attr_name=None)
        self.graph_paths = list(sorted(self.root.glob('*.pt'), key=lambda x: x.stem))

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, index):
        graph = torch.load(self.graph_paths[index])
        # get mask randomly
        mask_index = torch.randint(0, graph.concealer_masks.shape[0], (1,))[0]
        graph.concealer_mask = graph.concealer_masks[mask_index]
        del graph.concealer_masks
        # get label
        if self.task == 'concealer_size':
            graph = get_concealer_size(graph)
        elif self.task == 'binding_site':
            graph = get_binding_sites(graph)
        else:
            raise ValueError(self.task)
        del graph.concealer_mask

        graph.x = graph.x.to(torch.float32)
        graph.edge_index = graph.edge_index.to(torch.int64)
        graph.edge_attr = graph.edge_attr.to(torch.float32)
        graph.label = graph.label.to(torch.float32)

        if self.bidirection is not None:
            graph = self.bidirection(graph)

        if self.pos_enc is not None:
            graph = self.pos_enc(graph)
        return graph
