import dgl
import torch
import torch.nn.functional as F
from torch import nn


class HeteroRGATLayer(nn.Module):
    ''' Heterograph RGAT Layer Building Block
    '''

    def __init__(self, in_dim, out_dim, graph_relation_list):
        super(HeteroRGATLayer, self).__init__()
        # relation weight matrix
        self.relation_weight_matrix = nn.ModuleDict(
            {''.join(relation): nn.Linear(in_dim, out_dim) for relation in graph_relation_list})
        # attention weight matrix
        self.relation_attn_fc = nn.ModuleDict(
            {''.join(relation): nn.Linear(2 * out_dim, 1, bias=False) for relation in graph_relation_list})

    def forward(self, graph, node_feature_dict, eweight=None):
        ''' per relation message passing/reduction function dict
            relation => (message passing func, message reduction func)
        '''
        rel_func_dict = {}
        for relation in graph.canonical_etypes:
            # zero edges of this relation? we move onto next relation
            if not graph.num_edges(relation):
                continue
            relation_str = ''.join(relation)
            src = relation[0]
            dst = relation[2]
            udfFunctions = RelationUDF(relation_str, self.relation_attn_fc)

            # compute W_r * h
            wh_src = self.relation_weight_matrix[relation_str](node_feature_dict[src])
            wh_dst = self.relation_weight_matrix[relation_str](node_feature_dict[dst])

            # save in graph for message passing (z = whh'_)
            graph.nodes[src].data[f'wh_{relation_str}'] = wh_src
            graph.nodes[dst].data[f'wh_{relation_str}'] = wh_dst

            # edge weights support for GNNExplainer
            if eweight is not None:
                graph.edges[relation].data['w'] = eweight[relation].view(-1, 1)
            else:
                graph.edges[relation].data['w'] = torch.ones([graph.number_of_edges(relation), 1], device=graph.device)

            # equation (2)
            graph.apply_edges(udfFunctions.edge_attention, etype=relation)

            rel_func_dict[relation] = (udfFunctions.message_func, udfFunctions.reduce_func)

        # equation (3) & (4)
        # self.g.update_all(self.message_func, self.reduce_func)
        # trigger message passing & aggregation
        graph.multi_update_all(rel_func_dict, 'sum')

        # return self.g.ndata.pop('h')
        return {ntype: graph.nodes[ntype].data['h'] for ntype in graph.ntypes if graph.num_nodes(ntype)}


# Used for GAT Edge Attention, Message, and Reduce UDF Functions
class RelationUDF:

    def __init__(self, relation_str, attention_weight_matrix):
        self.relation_str = relation_str
        self.attention_weight_matrix = attention_weight_matrix

    def edge_attention(self, edges):
        ''' edge UDF for equation (2)
        '''
        wh2 = torch.cat([edges.src[f'wh_{self.relation_str}'], edges.dst[f'wh_{self.relation_str}']], dim=1)
        a = self.attention_weight_matrix[self.relation_str](wh2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        ''' message UDF for equation (3) & (4)
        '''
        return {f'wh_{self.relation_str}': edges.src[f'wh_{self.relation_str}'], 'e': edges.data['e'] * edges.data['w']}

    def reduce_func(self, nodes):
        ''' reduce UDF for equation (3) & (4)
        '''
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox[f'wh_{self.relation_str}'], dim=1)
        return {'h': h}


class KLayerHeteroRGAT(nn.Module):
    ''' K-Layer RGAT for Hetero-graph
        If num layers == 1, then hidden layer size does not matter and can be some arbitrary value
        structural: if we are training on graph structure & not considering any explicit node features
    '''

    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, graph_relation_list, graph_node_types, structural):
        super(KLayerHeteroRGAT, self).__init__()
        ''' embedding mapping for featureless heterograph
            graph memory location => embedding dict
        '''
        assert num_layers > 0, 'Number of layers in RGCN must be greater than 0!'
        self.input_feature_size = in_dim
        self.structural = structural

        if structural:
            # create shared node type features for ALL graphs
            feature_dict = {ntype: nn.Parameter(torch.zeros(1, in_dim)) for ntype in graph_node_types}
            for _, feature in feature_dict.items():
                nn.init.xavier_uniform_(feature)
            self.embed = nn.ParameterDict(feature_dict)

        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(HeteroRGATLayer(in_dim, out_dim, graph_relation_list))
        else:
            # Add first layer
            self.layers.append(HeteroRGATLayer(in_dim, hidden_dim, graph_relation_list))
            # Add intermediate layers
            for _ in range(0, num_layers - 2):
                self.layers.append(HeteroRGATLayer(hidden_dim, hidden_dim, graph_relation_list))
            # Add last (output layer)
            self.layers.append(HeteroRGATLayer(hidden_dim, out_dim, graph_relation_list))

    def forward(self, graph, feature_set=None, eweight=None):
        if not self.structural and feature_set is None:
            raise AssertionError('No feature set is given for given option of training GAT with non-structural data')

        if self.structural and feature_set is None:
            # Apply first layer
            x = self.layers[0](graph, {
                ntype: self.embed[ntype].expand(graph.num_nodes(ntype), -1) for ntype in self.embed
            },
                               eweight=eweight)
        else:
            x = self.layers[0](graph, feature_set, eweight=eweight)

        # Return w/o activation function if num layers == 1
        if len(self.layers) == 1:
            return x

        x = {node_type: F.elu(output) for node_type, output in x.items()}

        # Apply interned layers
        for i in range(1, len(self.layers) - 1):
            x = self.layers[i](graph, x, eweight=eweight)
            x = {node_type: F.elu(output) for node_type, output in x.items()}

        # Apply last layer w/o activation function
        x = self.layers[-1](graph, x, eweight)
        return x

    def get_node_features(self, graph):
        return {ntype: self.embed[ntype].expand(graph.num_nodes(ntype), -1) for ntype in self.embed}


# Graph Classifier for Prov Graph Dataset (ie benign or malicious)
class BinaryHeteroClassifier(nn.Module):

    def __init__(
        self,
        gnn_type,
        num_layers,
        input_feature_size,
        hidden_dimension_size,
        graph_relation_list,
        graph_node_types,
        structural=True,
    ):
        super(BinaryHeteroClassifier, self).__init__()
        assert gnn_type in ['sgat'], 'Supported GNN types are [sgat]'

        if gnn_type == 'sgat':
            self.gnn = KLayerHeteroRGAT(
                num_layers,
                input_feature_size,
                hidden_dimension_size,
                hidden_dimension_size,
                graph_relation_list,
                graph_node_types,
                structural=structural,
            )

        self.classifier = nn.Linear(hidden_dimension_size, 1)

    def forward(self, graph, feature_set=None, eweight=None):
        x = self.gnn(graph, feature_set, eweight=eweight)

        # classify graph as benign/malicious
        with graph.local_scope():
            graph.ndata['x'] = x
            hg = 0
            for ntype in graph.ntypes:
                if graph.num_nodes(ntype):
                    hg = hg + dgl.mean_nodes(graph, 'x', ntype=ntype)

            return torch.sigmoid(self.classifier(hg))  # output probability

    def get_node_features(self, graph):
        return self.gnn.get_node_features(graph)


# Get S-GAT Feedbacks
class SGATEvaluator():

    def __init__(self, dataset_name, bidirection=True, device='cpu', batch_size=32):
        self.dataset_name = dataset_name
        self.bidirection = bidirection
        self.device = device
        self.batch_size = batch_size

        if dataset_name == 'supply':
            from dataset.supply.parser import Parser
            self.input_dim = 8
            self.hidden_dim = 64
            self.num_layers = 5
        elif dataset_name == 'atlas':
            from dataset.atlas.parser import Parser
            self.input_dim = 8
            self.hidden_dim = 64
            self.num_layers = 5
        else:
            raise NotImplementedError(dataset_name)

        self.parser = Parser(dataset_name, bidirection=bidirection)

        self.ntype_list = self.parser.valid_ntypes
        if bidirection:
            self.relation_list = self.parser.bidirection_relations
        else:
            self.relation_list = self.parser.valid_relations

        self.ckpt_path = f'src/pids/sgat/ckpt/{dataset_name}.pt'

        self.model = BinaryHeteroClassifier(gnn_type='sgat',
                                            num_layers=self.num_layers,
                                            input_feature_size=self.input_dim,
                                            hidden_dimension_size=self.hidden_dim,
                                            graph_relation_list=self.relation_list,
                                            graph_node_types=self.ntype_list).to(device)
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=device))

        self.threshold = 0.5  # anomaly threshold

    def __call__(self, graphs):
        if not isinstance(graphs, list):
            graphs = [graphs]
        batches = [graphs[i:i + self.batch_size] for i in range(0, len(graphs), self.batch_size)]
        anomaly_scores = []
        for batch in batches:
            batched_graphs = dgl.batch(batch).to(self.device)
            anomaly_scores.extend(self.evaluate(batched_graphs))
        return anomaly_scores

    def evaluate(self, batched_graphs):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(batched_graphs)
        anomaly_scores = pred.cpu().view(-1).tolist()
        return anomaly_scores
