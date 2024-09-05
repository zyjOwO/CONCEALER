import warnings
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer

from src.predictor.data import AtlasDataModule, SupplyDataModule
from src.predictor.model import BindingSitePredictor, ConcealerSizePredictor

torch.set_float32_matmul_precision('high')
warnings.filterwarnings('once')
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


def test(task, dataset_name, pids_name, **kwargs):
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

    # 模型
    dims = {'node_dim': datamodule.node_dim, 'edge_dim': datamodule.edge_dim}
    if task == 'binding_site':
        model = BindingSitePredictor(config, dims, stats=datamodule.stats)
    elif task == 'concealer_size':
        model = ConcealerSizePredictor(config, dims)
    else:
        raise ValueError(task)

    # Trainer
    trainer = Trainer(
        max_epochs=train_config['num_epochs'],
        accelerator=train_config['accelerator'],
        devices=1,
        logger=False,
    )

    # Testing
    trainer.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path=config['ckpt_path'],
    )
    return


if __name__ == '__main__':
    dataset_name = 'supply'
    pids_name = 'fga'
    gnn_type = 'GCN'
    task = 'binding_site'
    test(
        task=task,
        dataset_name=dataset_name,
        pids_name=pids_name,
        gnn_type=gnn_type,
    )
