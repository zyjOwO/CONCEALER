import re
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch
from gensim.models import Word2Vec
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, one_hot

from blosc_compress import blosc_pkl_dump

# Node Types
ATLAS_NTYPES = [
    'process',
    'windows_process',
    'system32_process',
    'programfiles_process',
    'user_process',
    'file',
    'windows_file',
    'system32_file',
    'programfiles_file',
    'user_file',
    'combined_files',
    'ip_address',
    'domain_name',
    'web_object',
    'session',
    'connection',
]
# Edge Types
ATLAS_ETYPES = [
    'read',
    'write',
    'delete',
    'execute',
    'executed',
    'fork',
    'connect',
    'resolve',
    'web_request',
    'refer',
    'bind',
    'sock_send',
    'connected_remote_ip',
    'connected_session',
]
# Relations
ATLAS_RAW_RELATIONS = [  # for log->nx_graph
    ('ip_address', 'resolve', 'domain_name'),
    ('domain_name', 'web_request', 'web_object'),
    ('web_object', 'refer', 'web_object'),
    ('file', 'executed', 'process'),
    ('process', 'fork', 'process'),
    ('ip_address', 'connected_remote_ip', 'process'),
    ('ip_address', 'connected_remote_ip', 'connection'),
    ('ip_address', 'connected_session', 'session'),
    ('process', 'connect', 'connection'),
    ('session', 'sock_send', 'session'),
    ('process', 'bind', 'session'),
    ('process', 'read', 'file'),
    ('process', 'write', 'file'),
    ('process', 'delete', 'file'),
    ('process', 'execute', 'file'),
]
ATLAS_RELATIONS = [  # for nx_graph->pyg/dgl data
    ('file', 'executed', 'process'),
    ('process', 'fork', 'process'),
    ('domain_name', 'web_request', 'web_object'),
    ('web_object', 'refer', 'web_object'),
    ('programfiles_process', 'read', 'file'),
    ('programfiles_file', 'executed', 'programfiles_process'),
    ('process', 'fork', 'programfiles_process'),
    ('session', 'sock_send', 'session'),
    ('system32_process', 'connect', 'connection'),
    ('system32_file', 'executed', 'system32_process'),
    ('process', 'fork', 'system32_process'),
    ('system32_process', 'read', 'file'),
    ('system32_process', 'read', 'system32_file'),
    ('process', 'connect', 'connection'),
    ('programfiles_process', 'delete', 'combined_files'),
    ('programfiles_process', 'read', 'programfiles_file'),
    ('programfiles_process', 'read', 'user_file'),
    ('programfiles_process', 'write', 'combined_files'),
    ('system32_process', 'fork', 'programfiles_process'),
    ('system32_process', 'read', 'user_file'),
    ('system32_process', 'fork', 'system32_process'),
    ('system32_process', 'read', 'combined_files'),
    ('system32_process', 'write', 'user_file'),
    ('windows_process', 'read', 'user_file'),
    ('windows_file', 'executed', 'windows_process'),
    ('process', 'fork', 'windows_process'),
    ('programfiles_process', 'delete', 'user_file'),
    ('system32_process', 'read', 'programfiles_file'),
    ('system32_process', 'read', 'windows_file'),
    ('system32_process', 'execute', 'windows_file'),
    ('system32_process', 'execute', 'system32_file'),
    ('programfiles_process', 'write', 'user_file'),
    ('windows_process', 'read', 'windows_file'),
    ('windows_process', 'write', 'user_file'),
    ('system32_process', 'write', 'file'),
    ('system32_process', 'delete', 'windows_file'),
    ('process', 'delete', 'combined_files'),
    ('system32_process', 'write', 'windows_file'),
    ('system32_process', 'fork', 'process'),
    ('system32_process', 'write', 'combined_files'),
    ('system32_process', 'delete', 'file'),
    ('system32_process', 'delete', 'combined_files'),
    ('system32_process', 'execute', 'combined_files'),
    ('system32_process', 'execute', 'programfiles_file'),
    ('programfiles_process', 'execute', 'system32_file'),
    ('programfiles_process', 'read', 'system32_file'),
    ('programfiles_process', 'execute', 'programfiles_file'),
    ('programfiles_process', 'execute', 'windows_file'),
    ('programfiles_process', 'read', 'windows_file'),
    ('programfiles_process', 'read', 'combined_files'),
    ('system32_process', 'write', 'system32_file'),
    ('system32_process', 'fork', 'windows_process'),
    ('windows_process', 'read', 'combined_files'),
    ('windows_process', 'read', 'file'),
    ('windows_process', 'fork', 'system32_process'),
    ('windows_process', 'read', 'system32_file'),
    ('windows_process', 'execute', 'windows_file'),
    ('windows_process', 'read', 'programfiles_file'),
    ('windows_process', 'execute', 'system32_file'),
    ('windows_process', 'fork', 'programfiles_process'),
    ('windows_process', 'execute', 'programfiles_file'),
    ('programfiles_process', 'fork', 'programfiles_process'),
    ('programfiles_process', 'connect', 'connection'),
    ('windows_process', 'fork', 'process'),
    ('ip_address', 'resolve', 'domain_name'),
    ('system32_process', 'delete', 'system32_file'),
    ('programfiles_process', 'fork', 'process'),
    ('programfiles_process', 'fork', 'system32_process'),
    ('system32_process', 'delete', 'user_file'),
    ('process', 'read', 'user_file'),
    ('windows_process', 'write', 'combined_files'),
    ('system32_process', 'execute', 'user_file'),
    ('ip_address', 'connected_remote_ip', 'system32_process'),
    ('ip_address', 'connected_remote_ip', 'connection'),
    ('ip_address', 'connected_session', 'session'),
    ('system32_process', 'bind', 'session'),
    ('process', 'read', 'system32_file'),
    ('ip_address', 'connected_remote_ip', 'process'),
    ('process', 'bind', 'session'),
    ('ip_address', 'connected_remote_ip', 'programfiles_process'),
    ('programfiles_process', 'bind', 'session'),
    ('user_process', 'write', 'user_file'),
    ('user_process', 'read', 'system32_file'),
    ('user_process', 'read', 'programfiles_file'),
    ('user_process', 'read', 'windows_file'),
    ('user_process', 'execute', 'user_file'),
    ('user_process', 'read', 'combined_files'),
    ('system32_process', 'fork', 'user_process'),
    ('user_file', 'executed', 'user_process'),
    ('user_process', 'read', 'file'),
    ('ip_address', 'connected_remote_ip', 'user_process'),
    ('user_process', 'connect', 'connection'),
    ('user_process', 'execute', 'windows_file'),
    ('user_process', 'delete', 'combined_files'),
    ('user_process', 'read', 'user_file'),
    ('user_process', 'bind', 'session'),
    ('system32_process', 'execute', 'file'),
    ('process', 'read', 'windows_file'),
    ('process', 'execute', 'user_file'),
    ('process', 'execute', 'windows_file'),
    ('process', 'read', 'file'),
    ('process', 'write', 'combined_files'),
    ('process', 'read', 'combined_files'),
    ('windows_process', 'delete', 'combined_files'),
    ('process', 'delete', 'system32_file'),
    ('programfiles_process', 'write', 'file'),
    ('programfiles_process', 'write', 'windows_file'),
    ('programfiles_process', 'execute', 'combined_files'),
    ('process', 'execute', 'system32_file'),
    ('windows_process', 'write', 'file'),
    ('windows_process', 'delete', 'user_file'),
    ('user_process', 'fork', 'user_process'),
    ('user_process', 'write', 'combined_files'),
    ('user_process', 'fork', 'process'),
    ('windows_process', 'fork', 'user_process'),
    ('process', 'write', 'file'),
    ('programfiles_process', 'execute', 'user_file'),
    ('process', 'fork', 'user_process'),
    ('programfiles_process', 'fork', 'user_process'),
    ('windows_process', 'write', 'windows_file'),
    ('windows_process', 'write', 'system32_file'),
    ('process', 'write', 'user_file'),
    ('process', 'execute', 'file'),
    ('windows_process', 'fork', 'windows_process'),
    ('ip_address', 'connected_remote_ip', 'windows_process'),
    ('windows_process', 'connect', 'connection'),
    ('windows_process', 'bind', 'session'),
    ('user_process', 'fork', 'system32_process'),
    ('programfiles_process', 'execute', 'file'),
]
# MAPPING
ATLAS_MAP = {
    # process
    'process': 'a',
    'windows_process': 'b',
    'system32_process': 'c',
    'programfiles_process': 'd',
    'user_process': 'e',
    # file
    'file': 'f',
    'windows_file': 'g',
    'system32_file': 'h',
    'programfiles_file': 'i',
    'user_file': 'j',
    'combined_files': 'k',
    # web
    'ip_address': 'l',
    'domain_name': 'm',
    'web_object': 'n',
    'session': 'o',
    'connection': 'p',
    # edge types
    'read': 'q',
    'write': 'r',
    'delete': 's',
    'execute': 't',
    'executed': 'u',
    'fork': 'v',
    'connect': 'w',
    'resolve': 'x',
    'web_request': 'y',
    'refer': 'z',
    'bind': 'A',
    'sock_send': 'B',
    'connected_remote_ip': 'C',
    'connected_session': 'D',
}


def get_process_type(process_name):
    if 'c:/windows/system32' in process_name:
        return 'system32_process'
    elif 'c:/windows' in process_name:
        return 'windows_process'
    elif 'c:/programfiles' in process_name:
        return 'programfiles_process'
    elif 'c:/users' in process_name:
        return 'user_process'
    else:
        return 'process'


def get_file_type(file_name):
    if ';' not in file_name:
        if 'c:/windows/system32' in file_name:
            return 'system32_file'
        elif 'c:/windows' in file_name:
            return 'windows_file'
        elif 'c:/programfiles' in file_name:
            return 'programfiles_file'
        elif 'c:/users' in file_name:
            return 'user_file'
        else:
            return 'file'
    else:
        return 'combined_files'


def atlas_preprocess(raw_dir=None):
    r'''processed (.dot) -> nx_graph (.pkl)
    '''
    # 1. paths
    raw_dir = Path('dataset/atlas/raw') if raw_dir is None else raw_dir
    processed_dir = raw_dir.parent.joinpath('processed')
    benign_dir = processed_dir.joinpath('benign')
    anomaly_dir = processed_dir.joinpath('anomaly')
    if benign_dir.exists() and anomaly_dir.exists():
        print('Processing is bypassed as the processed dataset directory already exists.')
        return
    benign_dir.mkdir(parents=True)
    anomaly_dir.mkdir(parents=True)

    # 2. processing
    for graph_path in sorted(processed_dir.glob('*.dot'), key=lambda x: x.stem):
        # 2.1 read malicious labels
        splitted_filename = re.split(r'[-_]', graph_path.stem)
        if splitted_filename[-1] in ['windows', 'py']:
            malicious_path = raw_dir.joinpath(f'{splitted_filename[0]}/malicious_labels.txt')
        else:
            malicious_path = raw_dir.joinpath(f'{splitted_filename[0]}/{splitted_filename[-1]}/malicious_labels.txt')
        malicious_labels = []
        with malicious_path.open('r') as f:
            for line in f.readlines():
                malicious_labels.append(line.lower().rstrip())
        # 2.2 read .dot files
        nx_graph = nx.DiGraph()
        with graph_path.open('r') as f:
            id_map = {}
            for line in f.readlines():
                # items
                line = line.strip('\n').split(' ')
                src_name, etype, dst_name = line
                # node type
                for relation in ATLAS_RAW_RELATIONS:
                    if etype == relation[1]:
                        src_type, dst_type = relation[0], relation[2]
                if etype == 'connected_remote_ip':
                    dst_type = 'connection' if 'connection_' in dst_name else 'process'
                if src_type == 'process':
                    src_type = get_process_type(src_name)
                if src_type == 'file':
                    src_type = get_file_type(src_name)
                if dst_type == 'process':
                    dst_type = get_process_type(dst_name)
                if dst_type == 'file':
                    dst_type = get_file_type(dst_name)
                # node name
                src_name, dst_name = f'{src_type}_{src_name}', f'{dst_type}_{dst_name}'
                # remove self-loops
                if src_name == dst_name:
                    continue
                # add nodes into nx_graph
                if src_name not in id_map:
                    id_map[src_name] = len(id_map)
                    label = False
                    for ml in malicious_labels:
                        if ml in src_name:
                            label = True
                    nx_graph.add_node(id_map[src_name], ntype=ATLAS_NTYPES.index(src_type), label=label)
                if dst_name not in id_map:
                    id_map[dst_name] = len(id_map)
                    label = False
                    for ml in malicious_labels:
                        if ml in dst_name:
                            label = True
                    nx_graph.add_node(id_map[dst_name], ntype=ATLAS_NTYPES.index(dst_type), label=label)
                # add edge into nx_graph
                src_id, dst_id = id_map[src_name], id_map[dst_name]
                if not nx_graph.has_edge(src_id, dst_id) and not nx_graph.has_edge(dst_id, src_id):
                    nx_graph.add_edge(src_id, dst_id, etype=ATLAS_ETYPES.index(etype))
        # 2.3 save malicious graph
        malicious_nodes = set([node for node, ntype in nx_graph.nodes(data=True) if ntype['label']])
        blosc_pkl_dump(nx_graph, anomaly_dir.joinpath(graph_path.with_suffix('.blosc').name))
        # 2.4 save benign graph
        nx_graph.remove_nodes_from(malicious_nodes)
        if not nx.is_weakly_connected(nx_graph):
            largest_cc = max(nx.weakly_connected_components(nx_graph), key=len)
            nx_graph = nx.convert_node_labels_to_integers(nx_graph.subgraph(largest_cc).copy())
        blosc_pkl_dump(nx_graph, benign_dir.joinpath(graph_path.with_suffix('.blosc').name))

    # 3. Done
    print(f'Done. The processed Atlas Dataset directory is located at: {processed_dir}')
    return


class Parser():
    r'''pipelines:
        processed -> concealer              for training denoising model
        processed -> PIDS (sgat/fga/flash)  for training PIDS detectors
        concealer -> PIDS (sgat/fga/flash)  for testing generated examples
    '''

    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name

        if dataset_name == 'atlas':
            self.valid_ntypes = ATLAS_NTYPES
            self.valid_etypes = ATLAS_ETYPES
            self.valid_relations = ATLAS_RELATIONS
        else:
            raise NotImplementedError

        # Flash
        self.flash_word2vec = kwargs['word2vec'] if 'word2vec' in kwargs else None
        self.flash_pos_enc = kwargs['pos_enc'] if 'pos_enc' in kwargs else None

        # S-GAT
        self.gat_bidirection = kwargs['bidirection'] if 'bidirection' in kwargs else None
        if self.gat_bidirection is not None:
            self.bidirection_relations = self.valid_relations.copy()
            for relation in self.bidirection_relations:
                flipped_relation = relation[::-1]
                if flipped_relation not in self.bidirection_relations:
                    self.bidirection_relations.append(flipped_relation)

    def processed2pyg(self, G: nx.DiGraph):
        r'''nx_digraph->pyg_homograph
        '''
        data = from_networkx(G, ['ntype'], ['etype'])  # NOTE Node IDs will be reordered
        data.x = data.x.to(torch.float32).view(-1)
        data.edge_index = data.edge_index.to(torch.int64)
        data.edge_attr = data.edge_attr.to(torch.float32).view(-1)
        data.label = data.label.to(torch.bool)
        return data.sort()

    def processed2concealer(self, G: nx.DiGraph, non_edge: bool):
        r'''ntypes and etypes -> onehot-encoded vectors
            non_edge: add the specific edge type
        '''
        data = self.processed2pyg(G)
        x, edge_attr = data.x, data.edge_attr
        data.x = one_hot(x.long(), len(self.valid_ntypes), torch.float32)
        if non_edge:
            edge_attr = edge_attr + 1
            data.edge_attr = one_hot(edge_attr.long(), len(self.valid_etypes) + 1, torch.float32)
        else:
            data.edge_attr = one_hot(edge_attr.long(), len(self.valid_etypes), torch.float32)
        return data

    def processed2predictor(self, G: nx.DiGraph):
        # nx_digraph->pyg_homograph
        return self.processed2concealer(G, False)

    def processed2gat(self, G: nx.DiGraph, bidirection=None):
        # nx_digraph->dgl_heterograph
        data = self.processed2predictor(G)
        if bidirection is None:
            bidirection = self.gat_bidirection
        g, id_map = self.concealer2gat(data, non_edge=False, bidirection=bidirection)
        return g, id_map

    def processed2fga(self, G: nx.DiGraph):
        # nx_digraph->pyg_homograph
        return self.processed2predictor(G)

    def processed2flash(self, G: nx.DiGraph, word2vec: Word2Vec, pos_enc: object):
        # nx_digraph->pyg_homograph
        data = self.processed2predictor(G)
        data = self.concealer2flash(data, non_edge=False, word2vec=word2vec, pos_enc=pos_enc)
        return data

    def concealer2gat(self, data: Data, non_edge: bool, bidirection=None):
        r'''pyg_homograph->dgl_heterograph
            NOTE DGL internally decides a deterministic order for the same set of node types and canonical edge types, 
            which does not necessarily follow the order in data_dict.
        '''
        # bidirection relations
        if bidirection is None:
            bidirection = self.gat_bidirection
        if bidirection:
            _valid_relations = self.bidirection_relations
        else:
            _valid_relations = self.valid_relations
        # convert graph object
        id_map = {key: {} for key in self.valid_ntypes}  # homogenous node ID -> heterogenous node ID
        graph_data = {key: ([], []) for key in _valid_relations}
        x = data.x.argmax(dim=-1).tolist()
        edge_index = data.edge_index.t().tolist()
        edge_attr = data.edge_attr.argmax(dim=-1).tolist()  # onehot vector -> etype
        for idx, edge in enumerate(edge_index):
            if non_edge:
                if edge_attr[idx] == 0:
                    continue
                edge_attr[idx] = edge_attr[idx] - 1
            src_id, dst_id = edge
            etype = self.valid_etypes[edge_attr[idx]]
            if isinstance(etype, tuple):
                src_type, etype, dst_type = etype
                assert src_type == self.valid_ntypes[x[src_id]]
                assert dst_type == self.valid_ntypes[x[dst_id]]
            else:
                src_type = self.valid_ntypes[x[src_id]]
                dst_type = self.valid_ntypes[x[dst_id]]
            relation = (src_type, etype, dst_type)
            assert relation in self.valid_relations
            if src_id not in id_map[src_type]:
                id_map[src_type][src_id] = len(id_map[src_type])
            if dst_id not in id_map[dst_type]:
                id_map[dst_type][dst_id] = len(id_map[dst_type])
            graph_data[relation][0].append(id_map[src_type][src_id])
            graph_data[relation][1].append(id_map[dst_type][dst_id])
            if bidirection:
                flipped_relation = relation[::-1]
                graph_data[flipped_relation][1].append(id_map[src_type][src_id])
                graph_data[flipped_relation][0].append(id_map[dst_type][dst_id])
        num_nodes_dict = {key: len(id_map[key]) for key in self.valid_ntypes}
        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        return g, id_map

    def concealer_nid_mapping(self, g: dgl.DGLHeteroGraph, concealer_masks: list, id_map: dict):
        r'''dgl homogenous node ID -> pyg homogenous node ID
        '''
        inv_map = {}
        for ntype in id_map:
            inv_map[ntype] = {int(v): int(k) for k, v in id_map[ntype].items()}
        # dgl ntypes align with valid_ntypes
        ntype_numbers = []
        for ntype in g.ntypes:
            ntype_numbers.extend([self.valid_ntypes.index(ntype)] * g.num_nodes(ntype))
        _, indices = torch.sort(torch.tensor(ntype_numbers), stable=True)
        # inv_mapped_indices: the mapping between homogenous node IDs and homogenous node IDs
        raw_nid = 0
        inv_mapped_indices = torch.zeros(g.num_nodes(), dtype=torch.long)
        for ntype in self.valid_ntypes:
            for i in g.nodes(ntype).tolist():
                inv_mapped_indices[raw_nid] = inv_map[ntype][i]
                raw_nid += 1
        # update concealer_masks according to inv_mapped_indices
        if not isinstance(concealer_masks[0], list):
            concealer_masks = [concealer_masks]
        for mask in concealer_masks:
            if sum(mask) == 0:
                continue
            sorted_mask = torch.tensor(mask)[indices]
            inv_mapped_mask = torch.zeros_like(sorted_mask)
            inv_mapped_mask[inv_mapped_indices] = sorted_mask
            mask.clear()
            mask.extend(inv_mapped_mask.tolist())
        return concealer_masks

    def concealer2fga(self, data: Data, non_edge: bool):
        data = self.onehot2type(data, non_edge)
        data.x = one_hot(data.x.to(torch.int64), num_classes=len(self.valid_ntypes))
        data.edge_attr = one_hot(data.edge_attr.to(torch.int64), num_classes=len(self.valid_etypes))
        return data

    def onehot2type(self, data: Data, non_edge: bool):
        # onehot vectors -> types
        if non_edge:
            edges_type = data.edge_attr.argmax(dim=-1)
            mask = edges_type.nonzero().view(-1)
            data.edge_index = data.edge_index[:, mask]
            data.edge_attr = data.edge_attr[mask]
        data.x = data.x.argmax(dim=-1).to(torch.float32)
        data.edge_attr = data.edge_attr.argmax(dim=-1).to(torch.float32)
        if non_edge:
            data.edge_attr = data.edge_attr - 1.
        return data

    def get_docs(self, data: Data):
        docs = {}
        for eid in range(data.num_edges):
            src_id, dst_id = data.edge_index[:, eid]
            etype = self.valid_etypes[data.edge_attr[eid].long()]
            if isinstance(etype, tuple):
                etype = f'{etype[0]}-{etype[1]}-{etype[2]}'
            docs.setdefault(src_id.item(), []).append(etype)
            docs.setdefault(dst_id.item(), []).append(etype)
        docs = [value for _, value in sorted(docs.items())]
        return docs

    def infer(self, doc, word2vec=None, pos_enc=None, dims=40):
        if word2vec is None:
            word2vec = self.flash_word2vec
        if pos_enc is None:
            pos_enc = self.flash_pos_enc
        word_embeddings = [word2vec.wv[word] for word in doc if word in word2vec.wv]
        if not word_embeddings:
            return np.zeros(dims, dtype=np.float32)
        output_embedding = torch.from_numpy(np.array(word_embeddings)).to(torch.float32)
        if len(doc) < 100000:
            output_embedding = pos_enc.embed(output_embedding)
        output_embedding = output_embedding.detach().cpu().numpy()
        return np.mean(output_embedding, axis=0)

    def flash_embedding(self, data: Data, word2vec=None, pos_enc=None):
        r''' Flash node embedding
        '''
        if word2vec is None:
            word2vec = self.flash_word2vec
        if pos_enc is None:
            pos_enc = self.flash_pos_enc
        data.y = data.x.clone().view(-1).long()
        docs = self.get_docs(data)
        x = [self.infer(doc, word2vec, pos_enc) for doc in docs]
        data.x = torch.from_numpy(np.array(x)).to(torch.float32)
        data.edge_attr = data.edge_attr.unsqueeze(-1)
        return data

    def concealer2flash(self, data: Data, non_edge: bool, word2vec=None, pos_enc=None):
        if word2vec is None:
            word2vec = self.flash_word2vec
        if pos_enc is None:
            pos_enc = self.flash_pos_enc
        data = self.onehot2type(data, non_edge)
        data = self.flash_embedding(data, word2vec, pos_enc)
        return data

    def processed2wordvec(self, G: nx.DiGraph):
        docs = {}
        for u, v, etype in G.edges(data=True):
            etype = etype['etype']
            etype = self.valid_etypes[etype]
            if isinstance(etype, tuple):
                etype = f'{etype[0]}-{etype[1]}-{etype[2]}'
            docs.setdefault(u, []).append(etype)
            docs.setdefault(v, []).append(etype)
        docs = [value for _, value in sorted(docs.items())]
        return docs


if __name__ == '__main__':
    atlas_preprocess()
