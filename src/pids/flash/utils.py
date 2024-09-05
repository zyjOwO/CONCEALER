import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.utils import contains_isolated_nodes, unbatch


class GCN(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_channel=32):
        super().__init__()
        self.conv1 = SAGEConv(in_channel, hidden_channel, normalize=True)
        self.conv2 = SAGEConv(hidden_channel, out_channel, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        return x


class PositionalEncoder:

    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]


class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, path):
        self.epoch = 0
        self.path = path

    def on_epoch_end(self, model):
        model.save(self.path)
        self.epoch += 1


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print('Epoch #{} start'.format(self.epoch))

    def on_epoch_end(self, model):
        print('Epoch #{} end'.format(self.epoch))
        self.epoch += 1


# Get FLASH Feedbacks
class FlashEvaluator():

    def __init__(self, dataset_name, device='cpu', batch_size=8):
        self.dataset_name = dataset_name
        self.device = device
        self.batch_size = batch_size

        if dataset_name == 'supply':
            from dataset.supply.parser import Parser
            self.x_dim = 30
            self.hidden_channel = 128
        elif dataset_name == 'atlas':
            from dataset.atlas.parser import Parser
            self.x_dim = 40
            self.hidden_channel = 128
        else:
            raise NotImplementedError(dataset_name)

        self.gnn_ckpt_path = f'src/pids/flash/ckpt/{dataset_name}.pt'
        self.word2vec_ckpt_path = f'src/pids/flash/ckpt/{dataset_name}_w2v.pt'

        self.pos_enc = PositionalEncoder(self.x_dim)
        self.word2vec = Word2Vec.load(self.word2vec_ckpt_path)

        self.parser = Parser(dataset_name, word2vec=self.word2vec, pos_enc=self.pos_enc)

        self.model = GCN(self.x_dim, len(self.parser.valid_ntypes), self.hidden_channel).to(device)
        self.remove_isolated_nodes = RemoveIsolatedNodes()

        ckpt_dict = torch.load(self.gnn_ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt_dict['model'])
        self.threshold = ckpt_dict['threshold']
        assert self.threshold < 1.

    def __call__(self, graphs):
        if not isinstance(graphs, list):
            graphs = [graphs]
        for idx, data in enumerate(graphs):
            if contains_isolated_nodes(data.edge_index, data.num_nodes):
                data = self.remove_isolated_nodes(data)
            graphs[idx] = self.parser.concealer2flash(data,
                                                      non_edge=False,
                                                      word2vec=self.word2vec,
                                                      pos_enc=self.pos_enc)
        anomaly_scores = []
        batches = [graphs[i:i + self.batch_size] for i in range(0, len(graphs), self.batch_size)]
        for batch in batches:
            batch = Batch.from_data_list(batch).to(self.device)
            anomaly_scores.extend(self.evaluate(batch))
        return anomaly_scores

    def evaluate(self, batch):
        self.model.eval()
        anomaly_scores = []
        with torch.no_grad():
            preds = self.model(batch.x, batch.edge_index)
            preds = unbatch(preds.argmax(dim=-1), batch.batch)
            labels = unbatch(batch.y, batch.batch)
            num_nodes = torch.bincount(batch.batch)
            # get the num_nodes of original graph
            if hasattr(batch, 'addition_mask'):
                addition_masks = unbatch(batch.addition_mask, batch.batch)
                for mk_idx in range(len(addition_masks)):
                    num_nodes[mk_idx] -= addition_masks[mk_idx].sum()
            elif hasattr(batch, 'num_addition_nodes'):
                num_nodes -= batch.num_addition_nodes
            # get anomaly scores
            for idx, label in enumerate(labels):
                pred = preds[idx]
                score = sum(pred != label) / num_nodes[idx]  # score = num_anomaly_nodes / num_nodes
                anomaly_scores.append(score.cpu().item())
        return anomaly_scores
