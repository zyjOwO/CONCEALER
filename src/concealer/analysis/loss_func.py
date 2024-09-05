import torch
import torch.nn.functional as F


class TrainLoss():  # cross_entropy loss

    def __init__(self, dataset_stats, graph_weight, device='cuda'):
        super().__init__()
        self.device = device
        self.ntype_weight = dataset_stats['concealer_ntype_weight']
        self.etype_weight = dataset_stats['concealer_etype_weight']
        self.node_weight = torch.tensor(graph_weight, dtype=torch.float32, device=device)

    def __call__(self, pred, target):
        edge_index = target.edge_index
        batch = target.batch
        graph_names = target.graph_name.cpu().tolist()
        edge_batch = batch[edge_index[0]]
        loss = 0.
        x_loss = 0.
        e_loss = 0.
        for id in torch.unique(batch):
            # unbatch node_attrs
            x_pred = pred.x[batch == id]
            x_target = target.x[batch == id]
            # unbatch edge_attrs
            e_pred = pred.edge_attr[edge_batch == id]
            e_target = target.edge_attr[edge_batch == id]
            # consider only concealer component (concealer nodes and edges)
            node_mask = pred.concealer_node_mask[batch == id].squeeze(-1)
            edge_mask = pred.concealer_edge_mask[edge_batch == id].squeeze(-1)
            # (num_nodes, dx) -> (num_nodes, )
            x_pred = x_pred[node_mask, :]
            x_target = torch.argmax(x_target[node_mask, :], dim=-1)
            # (num_edges, de)
            e_pred = e_pred[edge_mask, :]
            e_target = torch.argmax(e_target[edge_mask, :], dim=-1)
            # loss
            ntype_weight = self.ntype_weight[graph_names[id]].to(device=self.device, dtype=torch.float32)
            etype_weight = self.etype_weight[graph_names[id]].to(device=self.device, dtype=torch.float32)
            x_loss += F.cross_entropy(x_pred, x_target, weight=ntype_weight)
            e_loss += F.cross_entropy(e_pred, e_target, weight=etype_weight)
            loss += self.node_weight * x_loss + (1 - self.node_weight) * e_loss

        node_mask = pred.concealer_node_mask.squeeze(-1)
        edge_mask = pred.concealer_edge_mask.squeeze(-1)
        x_pred = pred.x[node_mask, :]
        e_pred = pred.edge_attr[edge_mask, :]
        x_target = torch.argmax(target.x[node_mask, :], dim=-1)
        e_target = torch.argmax(target.edge_attr[edge_mask, :], dim=-1)

        return loss, x_loss, e_loss, x_pred, x_target, e_pred, e_target
