from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from torch_geometric.utils import (
    remove_isolated_nodes,
    sort_edge_index,
    to_scipy_sparse_matrix,
)
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from blosc_compress import blosc_pkl_dump
from src.concealer.analysis import TestMetrics, TrainLoss
from src.concealer.data.utils import (
    AtlasApplyNoise,
    add_context_reverse_edges,
    apply_noise,
)
from src.predictor import BindingSitePrediction, ConcealerSizePrediction

from .ResGatedGCN import ResGatedGCN


def custom_largest_weakly_cc(data: Data, connection='weak'):
    assert data.edge_index is not None

    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    _, component = sp.csgraph.connected_components(adj, connection=connection)

    _, count = np.unique(component, return_counts=True)
    subset_np = np.in1d(component, count.argsort()[-1:])
    subset = torch.from_numpy(subset_np)
    subset = subset.to(data.edge_index.device, torch.bool)

    return data.subgraph(subset), subset


class Diffusion(LightningModule):  # Concealer Component Diffusion

    def __init__(self, config, dataset_stats, dataset_name, pids_name, experiment_name):
        super().__init__()
        # parser
        if dataset_name == 'atlas':
            from dataset.atlas.parser import Parser
        elif dataset_name == 'supply':
            from dataset.supply.parser import Parser
        self.parser = Parser(dataset_name)

        # config
        self.model_config = config['model']
        self.train_config = config['train']
        self.test_config = config['test']
        self.diffusion_config = config['diffusion']
        self.dataset_name = dataset_name
        self.pids_name = pids_name
        self.experiment_name = experiment_name

        # model args
        gnn_type = self.model_config['gnn_type']
        node_dim = dataset_stats['ntype_weight'].shape[0]
        edge_dim = dataset_stats['etype_weight'].shape[0]
        if gnn_type == 'ResGatedGCN':
            model_args = {
                'num_layers': self.model_config['num_layers'],
                'node_in_channels': node_dim,
                'node_hidden_channels': self.model_config['node_hidden_channels'],
                'node_out_channels': node_dim,
                'edge_in_channels': edge_dim,
                'edge_hidden_channels': self.model_config['edge_hidden_channels'],
                'edge_out_channels': edge_dim,
                'add_self_loops': self.model_config['add_self_loops'],
                'residual': self.model_config['residual'],
                'edge_model': self.model_config['edge_model'],
                'add_mask_attr': self.model_config['add_mask_attr'],
                'pos_enc_dim': self.model_config['pos_enc_dim'] if self.model_config['pos_enc'] else 0,
            }
            self.model = ResGatedGCN(**model_args)
        else:
            raise NotImplementedError(gnn_type)

        # loss function
        self.loss_func = TrainLoss(dataset_stats=dataset_stats,
                                   graph_weight=self.train_config['graph_weight'],
                                   device='cuda' if self.train_config['accelerator'] == 'gpu' else 'cpu')

        # metrics (anomaly scores)
        self.metrics = TestMetrics(dataset_name, device='cuda', num_evasion=True)

        # binding site and concealer size predictors
        if dataset_name in ['supply', 'atlas']:
            self.binding_site_predictor = BindingSitePrediction(dataset_name=dataset_name,
                                                                pids_name=pids_name,
                                                                experiment_name=experiment_name,
                                                                non_edge=True,
                                                                mode='pretrained',
                                                                device='cpu',
                                                                ckpt_path=self.test_config['binding_site_ckpt'])
            self.concealer_size_predictor = ConcealerSizePrediction(dataset_name=dataset_name,
                                                                    pids_name=pids_name,
                                                                    experiment_name=experiment_name,
                                                                    non_edge=True,
                                                                    mode='gaussian_pretrained',
                                                                    device='cpu',
                                                                    ckpt_path=self.test_config['concealer_size_ckpt'])
            if dataset_name == 'atlas':
                self.atlas_apply_noise = AtlasApplyNoise()
        else:
            raise NotImplementedError(dataset_name)

        # bidirectional edges
        self.bidirected = add_context_reverse_edges if self.model_config['bidirected'] else None

        # position encoding
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
        # valuation metrics
        self.node_f1 = MulticlassF1Score(node_dim)
        self.edge_f1 = MulticlassF1Score(edge_dim)
        self.node_acc = MulticlassAccuracy(node_dim)
        self.edge_acc = MulticlassAccuracy(edge_dim)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(),
                                      lr=float(self.train_config['lr']),
                                      amsgrad=True,
                                      weight_decay=float(self.train_config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        ''' batch: batched pyg data
            batch_idx: current training step
        '''
        # model
        noisy_data, clean_data = batch
        noisy_data_hat = self.forward(noisy_data)
        loss, node_loss, edge_loss, _, _, _, _ = self.loss_func(noisy_data_hat, clean_data)
        metrics = {'loss': loss, 'node_loss': node_loss, 'edge_loss': edge_loss}

        # log
        batch_size = noisy_data.num_graphs
        self.log(
            name='train_loss',
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='train_node_loss',
            value=node_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='train_edge_loss',
            value=edge_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        return metrics

    def validation_step(self, batch, batch_idx):
        ''' batch: batched pyg data
            batch_idx: current training step
        '''
        # model
        noisy_data, clean_data = batch
        noisy_data_hat = self.forward(noisy_data)
        loss, node_loss, edge_loss, x_pred, x_target, e_pred, e_target = self.loss_func(noisy_data_hat, clean_data)
        metrics = {'loss': loss, 'node_loss': node_loss, 'edge_loss': edge_loss}

        # log
        batch_size = noisy_data.num_graphs
        # losses
        self.log(
            name='val_loss',
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='val_node_loss',
            value=node_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='val_edge_loss',
            value=edge_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        # acc and f1
        node_f1_score = self.node_f1(x_pred.softmax(dim=-1), x_target.to(torch.long))
        edge_f1_score = self.edge_f1(e_pred.softmax(dim=-1), e_target.to(torch.long))
        node_acc = self.node_acc(x_pred.softmax(dim=-1), x_target.to(torch.long))
        edge_acc = self.edge_acc(e_pred.softmax(dim=-1), e_target.to(torch.long))
        self.log(
            name='val_node_f1',
            value=node_f1_score,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='val_edge_f1',
            value=edge_f1_score,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='val_node_acc',
            value=node_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='val_edge_acc',
            value=edge_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        return metrics

    def test_step(self, batch, batch_idx):
        ''' batch: batched pyg data
            batch_idx: current training step
        '''
        data_list = Batch.to_data_list(batch.cpu())
        for data in data_list:
            data = data.sort()
            # adversarial graph generation
            examples = []
            nodes_added = [0]
            edges_modified = [0]
            examples_to_generate = self.train_config['examples_to_generate']
            pbar = tqdm(total=examples_to_generate, desc='# of examples')
            while examples_to_generate:
                noisy_data, _ = self.anomaly_data_noising(data.clone())
                evasive_data = self.forward(noisy_data.clone().to(self.device))
                evasive_data = self.evasive_data_processing(evasive_data.cpu(), noisy_data, data)
                if evasive_data is not None:
                    examples.append(evasive_data)
                    examples_to_generate -= 1
                    pbar.update(1)
                    nodes_added.append(abs(evasive_data.num_nodes - data.num_nodes))
                    edges_modified.append(evasive_data.num_edges - data.num_edges)

            num_classes = data.edge_attr.shape[1] - 1
            node_attr = data.x.to(torch.float32)
            edge_attr = torch.argmax(data.edge_attr.to(torch.float32), dim=1) - 1
            edge_attr = F.one_hot(edge_attr, num_classes=num_classes).to(torch.float32)
            examples.insert(0, Data(node_attr, data.edge_index, edge_attr))
            anomaly_scores, num_evasion = self.metrics(examples, non_edge=False)

            graph_name = data.graph_name.item() if isinstance(data.graph_name, torch.Tensor) else data.graph_name
            saved_dir = Path(f'log/examples/{self.experiment_name}/{graph_name}')
            if not saved_dir.exists():
                saved_dir.mkdir(parents=True)
            blosc_pkl_dump([examples, anomaly_scores], saved_dir.joinpath(f'{graph_name}.blosc'))

            csv_path = saved_dir.joinpath('anomaly_score.csv')
            anomaly_scores['nodes_added'] = nodes_added
            anomaly_scores['edges_modified'] = edges_modified
            df = pd.DataFrame(anomaly_scores)
            df = df.round(6)
            df = df.round({'nodes_added': 0})
            df = df.round({'edges_modified': 0})
            df.to_csv(str(csv_path), index=False)
            csv_path = saved_dir.joinpath('num_evasion.csv')
            df = pd.DataFrame(num_evasion, index=[0])
            df.to_csv(str(csv_path), index=False)

    def anomaly_data_noising(self, data, size_mode='auto'):
        if self.dataset_name == 'supply':
            # binding sites
            binding_sites = self.binding_site_predictor(data)
            if 1 not in binding_sites.unique():
                rand_idx = torch.randint(0, binding_sites.shape[0], (1,))[0]
                binding_sites[rand_idx] = 1 - binding_sites[rand_idx]
            if not torch.equal(binding_sites.unique(), torch.tensor([0, 1])):
                raise KeyError(f'{binding_sites.unique()}')
            # concealer size
            if size_mode == 'auto':
                concealer_size = self.concealer_size_predictor(data)
            elif size_mode == 'given':
                concealer_size = torch.as_tensor(np.ceil(data.num_nodes * 0.01), dtype=torch.int64)
            else:
                raise NotImplementedError(size_mode)
            # add Gaussian noise into concealer component
            data.concealer_mask = torch.tensor([0] * data.num_nodes + [1] * concealer_size, dtype=torch.bool)
            data.x = torch.vstack([data.x, torch.randn([concealer_size, data.x.shape[1]])])
            noisy_data, _ = apply_noise(data, binding_sites=binding_sites)
        else:
            raise NotImplementedError(self.dataset_name)

        noisy_data.x = noisy_data.x.to(torch.float32)
        noisy_data.edge_attr = noisy_data.edge_attr.to(torch.float32)
        noisy_data.edge_index = noisy_data.edge_index.to(torch.int64)

        if self.bidirected is not None:
            noisy_data = self.bidirected(noisy_data)
        if self.pos_enc is not None:
            noisy_data = self.pos_enc(noisy_data)

        return noisy_data, concealer_size

    def evasive_data_processing(self, evasive_data, noisy_data, data):
        concealer_node_mask = evasive_data.concealer_node_mask.view(-1)
        concealer_edge_mask = evasive_data.concealer_edge_mask.view(-1)
        edge_index = evasive_data.edge_index.t().tolist()
        ntypes = torch.argmax(noisy_data.x[:, :len(self.parser.valid_ntypes)], dim=-1)
        etypes = torch.argmax(noisy_data.edge_attr, dim=-1)  # 包含无边类型
        ntypes[concealer_node_mask] = torch.argmax(evasive_data.x, dim=-1)[concealer_node_mask]
        etypes[concealer_edge_mask] = torch.argmax(evasive_data.edge_attr, dim=-1)[concealer_edge_mask]

        #### edge processing ####

        # remove edges associated with invalid relations
        valid_ntypes = deepcopy(self.parser.valid_ntypes)
        valid_etypes = deepcopy(self.parser.valid_etypes)
        valid_relations = deepcopy(self.parser.valid_relations)

        for idx, relation in enumerate(valid_relations):
            src_type_idx = valid_ntypes.index(relation[0])
            etype_idx = valid_etypes.index(relation) if self.dataset_name == 'supply' else valid_etypes.index(relation[1])
            dst_type_idx = valid_ntypes.index(relation[2])
            valid_relations[idx] = (src_type_idx, etype_idx, dst_type_idx)

        for idx, edge in enumerate(edge_index):
            src_type, etype, dst_type = ntypes[edge[0]], etypes[idx], ntypes[edge[1]]
            if etype == 0:  # non-edge
                continue
            relation = (src_type, etype - 1, dst_type)
            if relation not in valid_relations:
                inv_relation = (dst_type, etype - 1, src_type)
                if inv_relation in valid_relations:  # flipped edge
                    inv_edge = [edge[1], edge[0]]
                    if inv_edge in edge_index:
                        etypes[idx] = 0
                    else:
                        edge_index[idx] = inv_edge
                else:
                    etypes[idx] = 0

        # remove edges associated with non-edge type
        edge_mask = torch.as_tensor(etypes).nonzero().squeeze(-1)
        edge_index = torch.as_tensor(edge_index)
        edge_index = edge_index[edge_mask]
        edge_attr = noisy_data.edge_attr
        edge_attr[concealer_edge_mask] = evasive_data.edge_attr[concealer_edge_mask]
        edge_attr = edge_attr[edge_mask]

        # process bidirectional edges
        edge_mask = torch.ones(edge_attr.shape[0], dtype=torch.bool)
        edge_index = edge_index.tolist()
        for idx, edge in enumerate(edge_index):
            inv_edge = [edge[1], edge[0]]
            if inv_edge in edge_index:
                target_idx = edge_index.index(inv_edge)
                if edge_mask[idx] != edge_mask[target_idx]:
                    continue
                if edge_attr[idx].max() < edge_attr[target_idx].max():
                    edge_mask[idx] = False
                else:
                    edge_mask[target_idx] = False

        edge_index = torch.as_tensor(edge_index, dtype=torch.int64)[edge_mask].t()
        edge_attr = F.one_hot(torch.argmax(edge_attr[edge_mask], dim=-1) - 1, num_classes=len(valid_etypes))
        edge_index, edge_attr = sort_edge_index(edge_index, edge_attr)

        #### node processing ####

        # remove isolated nodes
        edge_index, edge_attr, node_mask = remove_isolated_nodes(edge_index, edge_attr, num_nodes=ntypes.shape[0])
        if node_mask.sum():
            ntypes = ntypes[node_mask]
            node_attr = F.one_hot(ntypes, num_classes=len(valid_ntypes))
        else:
            return None

        node_attr = node_attr.to(torch.float32)
        edge_index = edge_index.to(torch.int64)
        edge_attr = edge_attr.to(torch.float32)
        evasive_data = Data(node_attr, edge_index, edge_attr)

        # get largest weakly-connected subgraph
        if self.dataset_name == 'supply':
            evasive_data, _ = custom_largest_weakly_cc(evasive_data)
            evasive_data = evasive_data.sort()
        return evasive_data

    def forward(self, noisy_data):
        noisy_data.x = noisy_data.x.to(torch.float32)
        noisy_data.edge_index = noisy_data.edge_index.to(torch.int64)
        noisy_data.edge_attr = noisy_data.edge_attr.to(torch.float32)
        noisy_data_hat = self.model(noisy_data)
        return noisy_data_hat
