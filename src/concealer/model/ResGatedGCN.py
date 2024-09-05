from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index
from torch_scatter import scatter


class ResGatedGCN(nn.Module):

    def __init__(self,
                 node_in_channels,
                 node_hidden_channels,
                 node_out_channels,
                 edge_in_channels,
                 edge_hidden_channels,
                 edge_out_channels,
                 num_layers,
                 pos_enc_dim=0,
                 add_self_loops=True,
                 dropout=0.2,
                 residual=True,
                 edge_model=True,
                 add_mask_attr=True):
        super().__init__()
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.residual = residual
        self.edge_model = edge_model
        self.add_mask_attr = add_mask_attr

        self.node_dim = deepcopy(node_in_channels)
        self.edge_dim = deepcopy(edge_in_channels)

        node_in_channels += pos_enc_dim

        if self.add_mask_attr:
            node_in_channels += 1
            edge_in_channels += 1

        self.node_embedding_in = nn.Linear(node_in_channels, node_hidden_channels)
        self.node_embedding_out = nn.Linear(node_hidden_channels, node_out_channels)

        self.edge_embedding_in = nn.Linear(edge_in_channels, edge_hidden_channels)
        self.edge_embedding_out = nn.Linear(edge_hidden_channels, edge_out_channels)

        self.node_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            node_layer = GatedGCNLayer(node_in_channels=node_hidden_channels,
                                       node_hidden_channels=edge_hidden_channels,
                                       node_out_channels=node_hidden_channels,
                                       edge_in_channels=edge_hidden_channels,
                                       edge_hidden_channels=edge_hidden_channels,
                                       edge_out_channels=edge_hidden_channels,
                                       add_self_loops=add_self_loops,
                                       dropout=dropout,
                                       residual=residual)
            self.node_layers.append(node_layer)
            if self.edge_model:
                edge_layer = EdgeLayer(node_channels=node_hidden_channels, edge_channels=edge_hidden_channels)
                self.edge_layers.append(edge_layer)

    def reset_parameters(self):
        self.node_embedding_in.reset_parameters()
        self.node_embedding_out.reset_parameters()
        self.edge_embedding_in.reset_parameters()
        self.edge_embedding_out.reset_parameters()
        for layer in self.node_layers:
            layer.reset_parameters()
        for layer in self.edge_layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_mask = data.concealer_node_mask
        edge_mask = data.concealer_edge_mask

        x_in = x[:, :self.node_dim]
        e_in = edge_attr[:, :self.edge_dim]

        if self.add_mask_attr:
            x = torch.hstack((x, node_mask.to(x.dtype)))
            edge_attr = torch.hstack((edge_attr, edge_mask.to(edge_attr.dtype)))

        x = self.node_embedding_in(x)
        edge_attr = self.edge_embedding_in(edge_attr)

        for idx, layer in enumerate(self.node_layers):
            x, edge_attr = layer(x, edge_index, edge_attr)
            if self.edge_model:
                edge_attr = self.edge_layers[idx](x, edge_index, edge_attr, edge_mask)

        x = self.node_embedding_out(x)
        edge_attr = self.edge_embedding_out(edge_attr)

        if self.residual:
            x = x + x_in
            edge_attr = edge_attr + e_in

        data.x = x
        data.edge_attr = edge_attr
        return data

    def decode(self, x, edge_label_index):
        return (x[edge_label_index[0]] * x[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, x):
        prob_adj = x @ x.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class GatedGCNLayer(MessagePassing):

    def __init__(self,
                 node_in_channels,
                 node_hidden_channels,
                 node_out_channels,
                 edge_in_channels,
                 edge_hidden_channels,
                 edge_out_channels,
                 add_self_loops=True,
                 use_bn=None,
                 dropout=0.2,
                 residual=True,
                 bias=True):
        super().__init__()
        assert node_hidden_channels == edge_hidden_channels
        assert edge_hidden_channels == edge_out_channels

        self.A = nn.Linear(node_in_channels, node_hidden_channels, bias=bias)
        self.B = nn.Linear(node_in_channels, node_hidden_channels, bias=bias)
        self.C = nn.Linear(edge_in_channels, edge_hidden_channels, bias=bias)
        self.D = nn.Linear(node_in_channels, node_hidden_channels, bias=bias)
        self.E = nn.Linear(node_in_channels, node_hidden_channels, bias=bias)

        self.node_out = nn.Linear(node_hidden_channels, node_out_channels, bias=bias)

        self.bn_node = nn.BatchNorm1d(node_out_channels)
        self.bn_edge = nn.BatchNorm1d(edge_out_channels)

        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def reset_parameters(self):
        self.A.reset_parameters()
        self.B.reset_parameters()
        self.C.reset_parameters()
        self.D.reset_parameters()
        self.E.reset_parameters()
        self.node_out.reset_parameters()

    def forward(self, x, edge_index, e):
        if self.residual:
            x_in = x
            e_in = e

        if self.add_self_loops:
            edge_index, e = remove_self_loops(edge_index, e)
            edge_index, e = add_self_loops(edge_index, e, fill_value=1., num_nodes=x.size(0))

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax)

        x = self.node_out(x)
        x = F.leaky_relu(self.bn_node(x), inplace=True)
        if self.dropout > 0.:
            x = F.dropout(x, self.dropout, training=self.training)

        e = F.leaky_relu(self.bn_edge(e), inplace=True)
        if self.dropout > 0.:
            e = F.dropout(e, self.dropout, training=self.training)

        if self.add_self_loops:
            _, e = remove_self_loops(edge_index, e)

        if self.residual:
            x = x + x_in
            e = e + e_in

        return x, e

    def message(self, Dx_i, Ex_j, Ce):
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        dim_size = Bx.shape[0]

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size, reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


class EdgeLayer(nn.Module):

    def __init__(self, node_channels, edge_channels, residual=True, dropout=0.2):
        super().__init__()

        self.A = nn.Linear(edge_channels, edge_channels, bias=True)
        self.B = nn.Linear(edge_channels, edge_channels, bias=True)
        self.C = nn.Linear(node_channels, node_channels, bias=True)
        self.D = nn.Linear(node_channels + edge_channels, edge_channels, bias=True)
        self.E = nn.Linear(edge_channels, edge_channels, bias=True)

        self.bn_edge = nn.BatchNorm1d(edge_channels)
        self.residual = residual
        self.dropout = dropout

    def reset_parameters(self):
        self.A.reset_parameters()
        self.B.reset_parameters()
        self.C.reset_parameters()
        self.D.reset_parameters()
        self.E.reset_parameters()

    def forward(self, x, edge_index, edge_attr, edge_mask):
        edge_mask = edge_mask.squeeze(-1)
        e_ij = edge_attr[edge_mask]

        if self.residual:
            e_in = e_ij

        inv_edge_index = torch.flip(edge_index[:, edge_mask], dims=[0])
        _, e_ji = sort_edge_index(inv_edge_index, e_ij)

        e_ij = self.A(e_ij)
        e_ji = self.B(e_ji)

        # 将目标节点特征添加进边特征中
        x_j = self.C(x[edge_index[1, edge_mask]])
        e_ij = torch.hstack([x_j, e_ij])
        e_ij = self.D(e_ij)
        e_ij = torch.mul(F.sigmoid(self.E(e_ij + e_ji)), e_ij)

        e_ij = F.leaky_relu(self.bn_edge(e_ij), inplace=True)
        if self.dropout > 0.:
            e_ij = F.dropout(e_ij, self.dropout, training=self.training)

        if self.residual:
            e_ij = e_in + e_ij

        edge_attr[edge_mask] = e_ij
        return edge_attr
