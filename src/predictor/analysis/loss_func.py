import torch
from torch import nn
import torch.nn.functional as F


class BindingSiteLoss():  # BCE Loss

    def __init__(self, stats):
        super().__init__()
        self.weights = [stats['non_binding_sites_weight'], stats['binding_sites_weight']]

    def __call__(self, pred, target, batch, graph_names):
        ''' pred (num_nodes, 2)
            target (num_nodes, )
        '''
        loss = 0.
        for id in torch.unique(batch):
            pr = pred[batch == id]
            ta = target[batch == id]
            name = graph_names[id]
            weight = torch.hstack([self.weights[0][name], self.weights[1][name]])
            loss += F.cross_entropy(pr, ta.long(), weight=weight.to(pred.device))
        return loss


class ConcealerSizeLoss():  # MSE Loss

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def __call__(self, pred, target):
        ''' pred: (bs, 1)
            target (bs, )
        '''
        return self.criterion(pred.view(-1), target.float())
