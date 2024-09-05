import time
import warnings
from pathlib import Path

import torch
import wandb
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from src.predictor.data import AtlasDataModule, SupplyDataModule
from src.predictor.model import BindingSitePredictor, ConcealerSizePredictor

torch.set_float32_matmul_precision('high')
warnings.filterwarnings('once')
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')


def train(task, dataset_name, pids_name, **kwargs):
    # config
    if pids_name == 'fga':
        config_path = Path(f'config/{dataset_name}/{pids_name}/{kwargs["gnn_type"]}/{task}.yaml')
    else:
        config_path = Path(f'config/{dataset_name}/{pids_name}/{task}.yaml')
    with config_path.open('r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    train_config = config['train']
    model_config = config['model']

    # datamodule
    trainset_dir = Path(f'dataset/{dataset_name}/graph_pruning/{pids_name}')
    if pids_name == 'fga':
        trainset_dir = trainset_dir.joinpath(kwargs['gnn_type'])
    assert trainset_dir.exists()
    if dataset_name == 'atlas':
        datamodule = AtlasDataModule(trainset_dir,
                                     task,
                                     batch_size=train_config['batch_size'],
                                     bidirection=model_config['bidirection'],
                                     pos_enc=model_config['pos_enc'],
                                     pos_enc_dim=model_config['pos_enc_dim'])
    elif dataset_name == 'supply':
        datamodule = SupplyDataModule(trainset_dir,
                                      task,
                                      batch_size=train_config['batch_size'],
                                      bidirection=model_config['bidirection'],
                                      pos_enc=model_config['pos_enc'],
                                      pos_enc_dim=model_config['pos_enc_dim'])
    else:
        raise NotImplementedError(dataset_name)

    # model
    dims = {'node_dim': datamodule.node_dim, 'edge_dim': datamodule.edge_dim}
    if task == 'binding_site':
        model = BindingSitePredictor(config, dims, stats=datamodule.stats)
    elif task == 'concealer_size':
        model = ConcealerSizePredictor(config, dims)
    else:
        raise ValueError(task)

    # experiment name
    start_time = time.strftime(r'%y-%m-%d %H:%M:%S')
    if pids_name == 'fga':
        experiment_name = f'{start_time}_{dataset_name}-{pids_name}-{kwargs["gnn_type"]}-{task}'
    else:
        experiment_name = f'{start_time}_{dataset_name}-{pids_name}-{task}'

    # wandb logger
    wandb.init(dir='log', project='CONCEALER', name=experiment_name)
    wandb_logger = WandbLogger()

    # callbacks
    callbacks = []
    ckpt_dir = Path(f'src/predictor/ckpt/{task}')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)
    ckpt_callback = ModelCheckpoint(dirpath=str(ckpt_dir.joinpath(experiment_name)),
                                    filename='{epoch:03d}_{val_loss:.8f}',
                                    monitor='val_loss',
                                    save_top_k=4,
                                    mode='min',
                                    every_n_epochs=1)
    callbacks.append(ckpt_callback)

    # Trainer
    trainer = Trainer(max_epochs=train_config['num_epochs'],
                      accelerator=train_config['accelerator'],
                      devices=1,
                      logger=wandb_logger,
                      callbacks=callbacks)

    # Training
    trainer.fit(model=model, datamodule=datamodule)
    return ckpt_callback.best_model_path


if __name__ == '__main__':
    dataset_name = 'supply'  # atlas or supply
    pids_name = 'fga'  # fga, flash, or sgat
    gnn_type = 'GATv2'  # for fga
    task = 'binding_site'  # binding_site or concealer_size
    train(
        task=task,
        dataset_name=dataset_name,
        pids_name=pids_name,
        gnn_type=gnn_type,
    )
