from src.pids.flash.pruner import pruner

if __name__ == '__main__':
    dataset_name = 'supply'
    pids_name = 'fga'
    gnn_type = 'GCN'
    pruner(dataset_name, pids_name, gnn_type)
