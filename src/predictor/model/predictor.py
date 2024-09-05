import math

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
)

from src.predictor.analysis import BindingSiteLoss, ConcealerSizeLoss

from .classifier import ResGatedGCN


class BindingSitePredictor(LightningModule):

    def __init__(self, config, dims, **kwargs):
        super().__init__()

        # config
        model_config = config['model']
        self.train_config = config['train']

        # model args
        model_args = {
            'in_channels': dims['node_dim'],
            'hidden_channels': model_config['hidden_channels'],
            'out_channels': 2,
            'edge_dim': dims['edge_dim'],
            'num_layers': model_config['num_layers'],
            'dropout': model_config['dropout'],
            'add_self_loops': model_config['add_self_loops'],
            'use_bn': model_config['use_bn'],
            'residual': model_config['residual'],
            'bias': model_config['bias'],
            'pos_enc_dim': model_config['pos_enc_dim'] if model_config['pos_enc'] else 0,
        }

        # model
        gnn_type = model_config['gnn_type']
        if gnn_type == 'ResGatedGCN':
            self.model = ResGatedGCN(**model_args)
        else:
            raise NotImplementedError(gnn_type)

        # loss function
        self.loss_func = BindingSiteLoss(kwargs['stats'])
        self.save_hyperparameters()

        # valuation metrics
        self.b_cm = BinaryConfusionMatrix()
        self.b_f1 = BinaryF1Score()
        self.b_auroc = BinaryAUROC()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(),
                                      lr=float(self.train_config['lr']),
                                      amsgrad=True,
                                      weight_decay=float(self.train_config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        r'''batch: batched pyg data
            batch_idx: current training step
        '''
        pred = self.forward(batch)
        loss = self.loss_func(pred, batch.label, batch.batch, batch.graph_name)

        batch_size = batch.num_graphs
        self.log(name='train_loss', value=loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        r'''batch: batched pyg data
            batch_idx: current training step
        '''
        pred = self.forward(batch)
        loss = self.loss_func(pred, batch.label, batch.batch, batch.graph_name)

        batch_size = batch.num_graphs
        self.log(
            name='val_loss',
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

        label = batch.label.view(-1).long()
        pred = pred.argmax(dim=-1).long()
        cm = self.b_cm(pred, label)
        bf = self.b_f1(pred, label)
        ba = self.b_auroc(pred, label)

        # confusion matrix
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        self.log(
            name='TNR',
            value=tn / (tn + fp),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='FPR',
            value=fp / (tn + fp),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='FNR',
            value=fn / (fn + tp),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='TPR (Recall)',
            value=tp / (fn + tp),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        # precision
        self.log(
            name='Precision',
            value=tp / (fp + tp),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        # f1 and auroc
        self.log(
            name='val_f1',
            value=bf,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='val_auroc',
            value=ba,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def forward(self, batch):
        # Batch
        x = batch.x.to(torch.float32)
        edge_index = batch.edge_index.to(torch.int64)
        edge_attr = batch.edge_attr.to(torch.float32)
        # Graph Model
        pred = self.model(x, edge_index, edge_attr)
        return pred


class ConcealerSizePredictor(LightningModule):

    def __init__(self, config, dims, **kwargs):
        super().__init__()

        # config
        model_config = config['model']
        self.train_config = config['train']

        # model args
        model_args = {
            'in_channels': dims['node_dim'],
            'hidden_channels': model_config['hidden_channels'],
            'out_channels': model_config['hidden_channels'],
            'edge_dim': dims['edge_dim'],
            'num_layers': model_config['num_layers'],
            'dropout': model_config['dropout'],
            'add_self_loops': model_config['add_self_loops'],
            'use_bn': model_config['use_bn'],
            'residual': model_config['residual'],
            'bias': model_config['bias'],
            'pos_enc_dim': model_config['pos_enc_dim'] if model_config['pos_enc'] else 0
        }

        # model
        gnn_type = model_config['gnn_type']
        if gnn_type == 'ResGatedGCN':
            self.model = ResGatedGCN(**model_args)
        else:
            raise NotImplementedError(gnn_type)

        # global pooling
        pool_name = model_config['global_pooling']
        if pool_name == 'mean':
            self.global_pooling = global_mean_pool
        elif pool_name == 'max':
            self.global_pooling = global_max_pool
        elif pool_name == 'add':
            self.global_pooling = global_add_pool
        else:
            raise NotImplementedError(pool_name)

        # MLP
        lin_hidden_channels = math.ceil(model_config['hidden_channels'] / 2)
        self.out_lin = nn.Sequential(
            nn.Linear(model_config['hidden_channels'], lin_hidden_channels),
            nn.ReLU(True),
            nn.Linear(lin_hidden_channels, 1),
        )

        # loss function
        self.loss_func = ConcealerSizeLoss()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(),
                                      lr=float(self.train_config['lr']),
                                      amsgrad=True,
                                      weight_decay=float(self.train_config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        r'''batch: batched pyg data
            batch_idx: current training step
        '''
        pred = self.forward(batch)
        loss = self.loss_func(pred, batch.label)
        batch_size = batch.num_graphs
        self.log(
            name='train_loss',
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        ''' batch: batched pyg data
            batch_idx: current training step
        '''
        pred = self.forward(batch)
        loss = self.loss_func(pred, batch.label)

        error = (pred.view(-1).round() - batch.label.float()).abs()
        average_error = error.mean()
        max_error = error.max()

        # log
        batch_size = batch.num_graphs
        self.log(
            name='val_loss',
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='average_error',
            value=average_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            name='max_error',
            value=max_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def forward(self, batch):
        # Batch
        x = batch.x.to(torch.float32)
        edge_attr = batch.edge_attr.to(torch.float32)
        edge_index = batch.edge_index.to(torch.int64)
        # Graph Model
        x = self.model(x, edge_index, edge_attr)
        # Global Pooling
        x = self.global_pooling(x, batch.batch)
        # MLP Classifier
        pred = self.out_lin(x)
        return pred
