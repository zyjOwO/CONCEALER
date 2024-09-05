import torch.nn as nn

from src.concealer.model import GatedGCNLayer


class ResGatedGCN(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 edge_dim,
                 num_layers=4,
                 dropout=0.2,
                 add_self_loops=True,
                 bias=True,
                 pos_enc_dim=0,
                 use_bn=None,
                 residual=None):
        super().__init__()
        in_channels += pos_enc_dim

        self.node_embedding_in = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.node_embedding_out = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.edge_embedding_in = nn.Linear(edge_dim, hidden_channels, bias=bias)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GatedGCNLayer(node_in_channels=hidden_channels,
                              node_hidden_channels=hidden_channels,
                              node_out_channels=hidden_channels,
                              edge_in_channels=hidden_channels,
                              edge_hidden_channels=hidden_channels,
                              edge_out_channels=hidden_channels,
                              add_self_loops=add_self_loops,
                              dropout=dropout))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.node_embedding_in.reset_parameters()
        self.node_embedding_out.reset_parameters()
        self.edge_embedding_in.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.node_embedding_in(x)
        edge_attr = self.edge_embedding_in(edge_attr)
        for conv in self.convs:
            x, edge_attr = conv(x, edge_index, edge_attr)
        x = self.node_embedding_out(x)
        return x
