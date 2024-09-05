import time
import warnings
from pathlib import Path

import torch
import wandb
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from src.concealer.data import AtlasDataModule, SupplyDataModule
from src.concealer.model.diffusion import Diffusion

torch.set_float32_matmul_precision('high')
warnings.filterwarnings('once')
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')


def train(dataset_name, pids_name, **kwargs):
    # config
    if pids_name == 'fga':
        config_path = Path(f'config/{dataset_name}/{pids_name}/{kwargs["gnn_type"]}/concealer.yaml')
    else:
        config_path = Path(f'config/{dataset_name}/{pids_name}/concealer.yaml')
    with config_path.open('r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    train_config = config['train']

    # datamodule
    trainset_dir = Path(f'dataset/{dataset_name}/graph_pruning/{pids_name}')
    if pids_name == 'fga':
        trainset_dir = trainset_dir.joinpath(kwargs['gnn_type'])
    assert trainset_dir.exists()
    if dataset_name == 'atlas':
        datamodule = AtlasDataModule(trainset_dir, config=config)
    elif dataset_name == 'supply':
        datamodule = SupplyDataModule(trainset_dir, config=config)
    else:
        raise NotImplementedError(dataset_name)

    # experiment name
    if pids_name == 'fga':
        experiment_name = f'{dataset_name}-{pids_name}-{kwargs["gnn_type"]}-concealer'
    else:
        experiment_name = f'{dataset_name}-{pids_name}-concealer'
    start_time = time.strftime(r'%y-%m-%d %H:%M:%S')  # 实验开始时间
    experiment_name = f'{start_time}_' + experiment_name

    # wandb logger
    wandb.init(dir='log', project='CONCEALER', name=experiment_name)
    wandb_logger = WandbLogger()

    # callbacks
    callbacks = []
    ckpt_dir = Path('src/concealer/ckpt')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)
    ckpt_callback = ModelCheckpoint(dirpath=str(ckpt_dir.joinpath(experiment_name)),
                                    filename='{epoch:03d}_{val_loss:.4f}',
                                    monitor='val_loss',
                                    save_top_k=4,
                                    mode='min',
                                    every_n_epochs=1)
    callbacks.append(ckpt_callback)

    # diffusion model
    model = Diffusion(config, datamodule.stats, dataset_name, pids_name, experiment_name)

    # trainer
    trainer = Trainer(max_epochs=train_config['num_epochs'],
                      accelerator=train_config['accelerator'],
                      devices=1,
                      logger=wandb_logger,
                      callbacks=callbacks)

    # training
    trainer.fit(model=model, datamodule=datamodule)
    return ckpt_callback.best_model_path


if __name__ == '__main__':
    dataset_name = 'supply'
    pids_name = 'fga'
    gnn_type = 'GATv2'
    train(dataset_name, pids_name, gnn_type=gnn_type)
