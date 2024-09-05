import time
import warnings
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer

from src.concealer.attack_result import attack_result
from src.concealer.data import AtlasDataModule, SupplyDataModule
from src.concealer.model.diffusion import Diffusion

torch.set_float32_matmul_precision('high')
warnings.filterwarnings('once')
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')


def test(dataset_name, pids_name, **kwargs):
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
    experiment_dir = Path(f'log/examples/{experiment_name}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True)
    print(experiment_name)

    # diffusion model
    model = Diffusion(
        config,
        datamodule.stats,
        dataset_name,
        pids_name,
        experiment_name,
    )

    # Trainer
    trainer = Trainer(
        max_epochs=train_config['num_epochs'],
        accelerator=train_config['accelerator'],
        logger=False,
    )

    # testing
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=config['test']['concealer_ckpt'],
    )

    # attack results
    attack_result(experiment_dir, num_lowest=0., mode='simple')
    attack_result(experiment_dir, num_lowest=0.1, mode='simple')
    return


if __name__ == '__main__':
    dataset_name = 'supply'
    pids_name = 'fga'
    gnn_type = 'GATv2'
    test(dataset_name, pids_name, gnn_type=gnn_type)
