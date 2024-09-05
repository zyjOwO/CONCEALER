from pathlib import Path

import torch
import yaml
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
    ToUndirected,
)

from .model.predictor import BindingSitePredictor, ConcealerSizePredictor


class BindingSitePrediction():

    def __init__(self,
                 dataset_name='supply',
                 pids_name='sgat',
                 experiment_name=None,
                 non_edge=False,
                 mode='pretrained',
                 device='cpu',
                 **kwargs):
        super().__init__()
        self.mode = mode
        if mode == 'pretrained':
            # model
            ckpt_path = kwargs['ckpt_path']
            if dataset_name == 'supply' and pids_name == 'flash':
                raise NotImplementedError('flash is failed in supply dataset')
            self.model = BindingSitePredictor.load_from_checkpoint(checkpoint_path=ckpt_path).to(device)

            # config
            if pids_name == 'fga':
                if 'GCN' in experiment_name:
                    config_path = Path(f'config/{dataset_name}/{pids_name}/GCN/binding_site.yaml')
                elif 'GATv2' in experiment_name:
                    config_path = Path(f'config/{dataset_name}/{pids_name}/GATv2/binding_site.yaml')
                else:
                    raise NotImplementedError
            else:
                config_path = Path(f'config/{dataset_name}/{pids_name}/binding_site.yaml')
            with config_path.open('r') as f:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)

            model_config = config['model']
            bidirection = model_config['bidirection']
            pos_enc = model_config['pos_enc']
            pos_enc_dim = model_config['pos_enc_dim']

            self.bidirection = ToUndirected() if bidirection else None

            if pos_enc == 'laplacian':
                self.pos_enc = AddLaplacianEigenvectorPE(k=pos_enc_dim, attr_name=None, is_undirected=bidirection)
            elif pos_enc == 'random_walk':
                self.pos_enc = AddRandomWalkPE(walk_length=pos_enc_dim, attr_name=None)
            else:
                self.pos_enc = None
            self.non_edge = non_edge

    def __call__(self, data):
        if self.mode == 'pretrained':
            return self.forward(data)
        else:
            return self.mode

    def forward(self, data):
        data = data.clone()
        if self.non_edge:
            data.edge_attr = data.edge_attr[:, 1:]
        if self.bidirection is not None:
            data = self.bidirection(data)
        if self.pos_enc is not None:
            data = self.pos_enc(data)
        pred = self.model(data)
        pred = torch.argmax(pred, dim=-1)
        return pred


class ConcealerSizePrediction():

    def __init__(self,
                 dataset_name='supply',
                 pids_name='sgat',
                 experiment_name=None,
                 non_edge=False,
                 mode='gaussian_pretrained',
                 device='cpu',
                 **kwargs):
        super().__init__()
        self.mode = mode

        if mode in ['pretrained', 'gaussian_pretrained']:
            # model
            ckpt_path = kwargs['ckpt_path']
            if dataset_name == 'supply' and pids_name == 'flash':
                raise NotImplementedError('flash is failed in supply dataset')
            self.model = ConcealerSizePredictor.load_from_checkpoint(checkpoint_path=ckpt_path).to(device)

            # config
            if pids_name == 'fga':
                if 'GCN' in experiment_name:
                    config_path = Path(f'config/{dataset_name}/{pids_name}/GCN/concealer_size.yaml')
                elif 'GATv2' in experiment_name:
                    config_path = Path(f'config/{dataset_name}/{pids_name}/GATv2/concealer_size.yaml')
                else:
                    raise NotImplementedError
            else:
                config_path = Path(f'config/{dataset_name}/{pids_name}/concealer_size.yaml')
            with config_path.open('r') as f:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)

            model_config = config['model']
            bidirection = model_config['bidirection']
            pos_enc = model_config['pos_enc']
            pos_enc_dim = model_config['pos_enc_dim']

            self.bidirection = ToUndirected() if bidirection else None

            if pos_enc == 'laplacian':
                self.pos_enc = AddLaplacianEigenvectorPE(k=pos_enc_dim, attr_name=None, is_undirected=bidirection)
            elif pos_enc == 'random_walk':
                self.pos_enc = AddRandomWalkPE(walk_length=pos_enc_dim, attr_name=None)
            else:
                self.pos_enc = None

            self.non_edge = non_edge

    def __call__(self, data, existed_concealer_size=None):
        if self.mode == 'pretrained':
            sample = self.forward(data)[0]
            sample = sample.floor() if sample > 1 else sample.new_ones(1)[0]
            sample = sample.to(torch.int64)
            if existed_concealer_size is not None:
                if sample <= existed_concealer_size:
                    sample = sample.new_tensor(3)
                else:
                    sample -= existed_concealer_size
        elif self.mode == 'gaussian_pretrained':
            mu = self.forward(data)[0]
            if existed_concealer_size is not None:
                if mu <= existed_concealer_size:
                    mu = mu.new_tensor(3)
                else:
                    mu -= existed_concealer_size
            diff = mu if mu <= (data.num_nodes - mu).abs() else (data.num_nodes - mu).abs()
            sigma = diff / 2
            sample = torch.normal(mean=mu, std=sigma).clamp(min=1, max=int(mu + diff))
            sample = sample.floor() if sample > 1 else sample.new_ones(1)[0]
            sample = sample.to(torch.int64)
        else:  # random
            max_num_concealer = torch.ceil(data.num_nodes * 0.1).clamp(min=1)
            sample = torch.randint(1, max_num_concealer, (1,), dtype=torch.int64)[0]
            if existed_concealer_size is not None:
                if sample <= existed_concealer_size:
                    sample = sample.new_ones(1)
                else:
                    sample -= existed_concealer_size
        assert sample >= 1
        if sample >= data.num_nodes:
            sample = torch.tensor(data.num_nodes - 1, dtype=sample.dtype)
        return sample

    def forward(self, data):
        data = data.clone()
        if self.non_edge:
            data.edge_attr = data.edge_attr[:, 1:]
        if self.bidirection is not None:
            data = self.bidirection(data)
        if self.pos_enc is not None:
            data = self.pos_enc(data)
        pred = self.model(data)
        pred = pred.view(-1).ceil().relu() + 1
        return pred
