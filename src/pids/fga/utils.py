import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import ARGVA, GATv2Conv, GCNConv
from torch_geometric.utils import unbatch


class Encoder(nn.Module):  # GCN Layer

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):  # no edge_attr
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class GATv2Encoder(nn.Module):  # GATv2 Layer

    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):  # edge_attr
        x = self.conv1(x, edge_index, edge_attr).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)


class AutoencoderEvaluator:

    def __init__(self, dataset_name, gnn_type, device='cpu', batch_size=8):
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.device = device
        self.batch_size = batch_size

        if dataset_name == 'supply':
            from dataset.supply.parser import Parser
            self.parser = Parser(dataset_name)
            self.encoder_input = len(self.parser.valid_ntypes)
            self.encoder_hidden = 128
            self.encoder_output = 32
            self.encoder_edge_dim = len(self.parser.valid_etypes)
            self.discriminator_input = 32
            self.discriminator_hidden = 128
            self.discriminator_out = 32
        elif dataset_name == 'atlas':
            from dataset.atlas.parser import Parser
            self.parser = Parser(dataset_name)
            self.encoder_input = len(self.parser.valid_ntypes)
            self.encoder_hidden = 128
            self.encoder_output = 32
            self.encoder_edge_dim = len(self.parser.valid_etypes)
            self.discriminator_input = 32
            self.discriminator_hidden = 128
            self.discriminator_out = 32
        else:
            raise NotImplementedError(dataset_name)

        self.ckpt_path = f'src/pids/fga/ckpt/{dataset_name}-{gnn_type}.pt'
        self.embedding_path = f'src/pids/fga/ckpt/{dataset_name}-{gnn_type}-benign-embedding.pt'

        if gnn_type == 'GCN':
            self.encoder = Encoder(self.encoder_input, self.encoder_hidden, self.encoder_output)
        elif gnn_type == 'GATv2':
            self.encoder = GATv2Encoder(self.encoder_input, self.encoder_hidden, self.encoder_output, self.encoder_edge_dim)
        else:
            raise ValueError(gnn_type)

        self.discriminator = Discriminator(in_channels=self.discriminator_input,
                                           hidden_channels=self.discriminator_hidden,
                                           out_channels=self.discriminator_out).to(device)

        self.model = ARGVA(self.encoder, self.discriminator).to(device)
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=device))

        embeddings_dict = torch.load(self.embedding_path, map_location=device)
        self.benign_embeddings = embeddings_dict['embeddings']
        self.threshold = embeddings_dict['threshold'].cpu()  # anomaly threshold

    def __call__(self, graphs):
        if not isinstance(graphs, list):
            graphs = [graphs]
        anomaly_scores = []
        batches = [graphs[i:i + self.batch_size] for i in range(0, len(graphs), self.batch_size)]
        for batch in batches:
            batch = Batch.from_data_list(batch).to(self.device)
            anomaly_scores.extend(self.evaluate(batch))
        return anomaly_scores

    def evaluate(self, batch):
        self.model.eval()
        with torch.no_grad():
            x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
            if self.gnn_type == 'GCN':
                z = self.model.encode(x, edge_index)
            elif self.gnn_type == 'GATv2':
                z = self.model.encode(x, edge_index, edge_attr)
            else:
                raise ValueError(self.gnn_type)
            anomaly_embeddings = None
            for item in unbatch(z, batch.batch):
                item = item.unsqueeze(0).detach()
                item = torch.mean(item, dim=1)
                if anomaly_embeddings is None:
                    anomaly_embeddings = item
                else:
                    anomaly_embeddings = torch.vstack([anomaly_embeddings, item])
        anomaly_scores = torch.min(torch.cdist(self.benign_embeddings, anomaly_embeddings), dim=0).values
        return anomaly_scores.cpu().tolist()


if __name__ == '__main__':
    from pathlib import Path
    dataset = 'supply'
    gnn_type = 'GCN'
    evaluator = AutoencoderEvaluator(dataset, gnn_type)
    parsed_dir = Path(f'dataset/{dataset}/pids/fga')
    pyg_benign_path = parsed_dir.joinpath('benign_graphs.pt')
    benign_graphs = torch.load(pyg_benign_path)
    print(f'benign_graph_scores: {evaluator(benign_graphs)}')
    pyg_anomaly_path = parsed_dir.joinpath('anomaly_graphs.pt')
    anomaly_graphs = torch.load(pyg_anomaly_path)
    print(f'anomaly_graph_scores: {evaluator(anomaly_graphs)}')
