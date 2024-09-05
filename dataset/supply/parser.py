import tarfile
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl import DGLHeteroGraph
from gensim.models import Word2Vec
from networkx import DiGraph
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, one_hot
from tqdm import tqdm

from blosc_compress import blosc_pkl_dump

# Node Types
SUPPLY_NTYPES = ['ProcessNode', 'SocketChannelNode', 'FileNode']

# Relations
SUPPLY_RELATIONS = [('ProcessNode', 'PROC_CREATE', 'ProcessNode'), ('ProcessNode', 'READ', 'FileNode'),
                    ('ProcessNode', 'WRITE', 'FileNode'), ('ProcessNode', 'FILE_EXEC', 'FileNode'),
                    ('ProcessNode', 'WRITE', 'SocketChannelNode'), ('ProcessNode', 'READ', 'SocketChannelNode'),
                    ('ProcessNode', 'IP_CONNECTION_EDGE', 'ProcessNode'),
                    ('ProcessNode', 'IP_CONNECTION_EDGE', 'FileNode')]

# Ignored Files
IGNORED_CSV_FILES = ['scoring_stats.csv', 'graph_properties.csv', 'file.csv', 'process.csv', 'path_score.csv']


# Data Preprocessing
def supply_preprocess(raw_dir=None):
    r'''raw_file -> nx_digraph (directed acyclic graph) -> largest weakly-connected subgraph -> *.blosc file
        HACK Low execution speed
    '''
    # 1. paths
    raw_dir = Path('dataset/supply/raw') if raw_dir is None else raw_dir
    processed_dir = raw_dir.parent.joinpath('processed')
    if not processed_dir.joinpath('graph_names.txt').exists():
        print(f'Processing SupplyChain Dataset... The dataset directory is located at: {raw_dir}')
    else:
        print('Processing is bypassed as the processed dataset directory already exists.')
        return

    # 2. file decompression
    if not raw_dir.joinpath('benign').exists():
        tar_filename = 'supply_raw_data.tar.gz'
        tar_path = raw_dir.joinpath(tar_filename)
        if not tar_path.exists():
            raise ValueError('the raw compressed file does not exist.')
        with tarfile.open(tar_path, 'r:gz') as tar_file:
            tar_file.extractall(raw_dir, filter='fully_trusted')

    # 3. processing
    graph_names = []
    for glabel_dir in raw_dir.iterdir():
        if not glabel_dir.is_dir() or glabel_dir.name not in ['benign', 'anomaly']:
            continue
        file_iter = sorted(glabel_dir.iterdir(), key=lambda x: x.stem)
        pbar = tqdm(file_iter, desc=f'process {glabel_dir.name} files', leave=False)
        for graph_dir in pbar:
            if not graph_dir.is_dir():
                continue
            # 3.1 read relation csv files
            skip_flag = False
            relation_files = []
            for csv_file in graph_dir.glob('*.csv'):
                if csv_file.name in IGNORED_CSV_FILES:
                    continue
                if '~' in csv_file.stem:
                    relation = tuple(csv_file.stem.split('~'))
                    if relation[0] != 'ProcessNode' or relation not in SUPPLY_RELATIONS:
                        tqdm.write(f'The {graph_dir.name} was skipped because invalid relation {relation} exists.')
                        skip_flag = True
                        break
                    else:
                        relation_files.append(csv_file)
            if skip_flag:
                continue
            else:
                graph_names.append(graph_dir.name)

            # 3.2 processed filepaths
            glabel_processed_dir = Path(str(glabel_dir).replace('raw', 'processed'))
            if not glabel_processed_dir.exists():
                glabel_processed_dir.mkdir(parents=True)
            graph_id = len(graph_names) - 1  # convert graph names to ID numbers
            graph_path = glabel_processed_dir.joinpath(f'{graph_id}.txt')

            # 3.3 convert raw graph（heterogeneous graph) to edge lists
            with graph_path.open('a', encoding='UTF8') as fw:
                for f in relation_files:
                    relation = tuple(f.stem.split('~'))
                    df = pd.read_csv(f)
                    # source node ID and destination node ID
                    src_ids = df['u'].to_list()
                    dst_ids = df['v'].to_list()
                    # ntypes
                    src_types = [relation[0]] * len(src_ids)
                    dst_types = [relation[2]] * len(dst_ids)
                    # etypes (i.e., relations)
                    etype = f'{relation[0]}-{relation[1]}-{relation[2]}'
                    etypes = [str(etype)] * len(src_ids)
                    # save edge lists to *.txt file
                    lines = np.array([src_ids, src_types, dst_ids, dst_types, etypes])
                    lines = lines.transpose().tolist()
                    for line in lines:
                        fw.write('\t'.join(line) + '\n')

            # 3.4 edge_lists -> nx_digraph -> remove self-loop and parallel edges -> largest weakly-connected subgraph -> *.blosc file
            nx_graph = DiGraph()
            with graph_path.open('r') as f:
                id_map = {}  # covert heterogenous node ID to homogenous one
                for line in f.readlines():
                    line = line.strip('\n').split('\t')
                    src_id, src_type, dst_id, dst_type, etype = line
                    src_id = f'{src_type}_{src_id}'
                    dst_id = f'{dst_type}_{dst_id}'
                    etype = tuple(etype.split('-'))
                    if src_type not in SUPPLY_NTYPES or dst_type not in SUPPLY_NTYPES:
                        continue
                    if etype not in SUPPLY_RELATIONS:
                        continue
                    if src_id == dst_id:  # remove self-loop
                        continue
                    if src_id not in id_map:
                        id_map[src_id] = len(id_map)
                        nx_graph.add_node(id_map[src_id], ntype=SUPPLY_NTYPES.index(src_type))
                    if dst_id not in id_map:
                        id_map[dst_id] = len(id_map)
                        nx_graph.add_node(id_map[dst_id], ntype=SUPPLY_NTYPES.index(dst_type))
                    src_id = id_map[src_id]
                    dst_id = id_map[dst_id]
                    # remove parallel edges
                    if not nx_graph.has_edge(src_id, dst_id) and not nx_graph.has_edge(dst_id, src_id):
                        nx_graph.add_edge(src_id, dst_id, etype=SUPPLY_RELATIONS.index(etype))
            # largest weakly-connected subgraph
            if not nx.is_weakly_connected(nx_graph):
                largest_cc = max(nx.weakly_connected_components(nx_graph), key=len)
                nx_graph = nx.convert_node_labels_to_integers(nx_graph.subgraph(largest_cc).copy())
            # save nx_digraph to *.blosc file
            graph_path.unlink()
            with graph_path.with_suffix('.blosc').open('wb') as fw:
                blosc_pkl_dump(nx_graph, fw)

    # 4. save original graph names
    name_path = graph_path.parents[1].joinpath('graph_names.txt')
    with name_path.open('a', encoding='UTF8') as fw:
        for name in graph_names:
            fw.write(str(name) + '\n')

    # 5. Done
    print(f'Done. The processed SupplyChain Dataset directory is located at: {processed_dir}')
    return


# Parser
class Parser():
    r'''pipelines:
        processed -> concealer              for training denoising model
        processed -> PIDS (sgat/fga/flash)  for training PIDS detectors
        concealer -> PIDS (sgat/fga/flash)  for testing generated examples
    '''

    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name

        if dataset_name == 'supply':
            self.valid_ntypes = SUPPLY_NTYPES
            self.valid_etypes = SUPPLY_RELATIONS
            self.valid_relations = SUPPLY_RELATIONS
        else:
            raise NotImplementedError(dataset_name)

        # FLASH
        self.flash_word2vec = kwargs['word2vec'] if 'word2vec' in kwargs else None
        self.flash_pos_enc = kwargs['pos_enc'] if 'pos_enc' in kwargs else None

        # S-GAT
        self.sgat_bidirection = kwargs['bidirection'] if 'bidirection' in kwargs else None
        self.bidirection_relations = self.valid_relations.copy()
        for relation in self.bidirection_relations:
            flipped_relation = relation[::-1]
            if flipped_relation not in self.bidirection_relations:
                self.bidirection_relations.append(flipped_relation)

    def processed2pyg(self, G: DiGraph):
        r'''nx_digraph->pyg_homograph
        '''
        data = from_networkx(G, ['ntype'], ['etype'])  # NOTE Node IDs will be reordered
        data.x = data.x.to(torch.float32).view(-1)
        data.edge_index = data.edge_index.to(torch.int64)
        data.edge_attr = data.edge_attr.to(torch.float32).view(-1)
        return data.sort()

    def processed2concealer(self, G: DiGraph, non_edge: bool):
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

    def processed2predictor(self, G: DiGraph):
        # nx_digraph->pyg_homograph
        return self.processed2concealer(G, False)

    def processed2sgat(self, G: DiGraph, bidirection=None):
        # nx_digraph->dgl_heterograph
        data = self.processed2predictor(G)
        if bidirection is None:
            bidirection = self.sgat_bidirection
        g, id_map = self.concealer2sgat(data, non_edge=False, bidirection=bidirection)
        return g, id_map

    def processed2fga(self, G: DiGraph):
        # nx_digraph->pyg_homograph
        return self.processed2predictor(G)

    def processed2flash(self, G: DiGraph, word2vec: Word2Vec, pos_enc: object):
        # nx_digraph->pyg_homograph
        data = self.processed2predictor(G)
        data = self.concealer2flash(data, non_edge=False, word2vec=word2vec, pos_enc=pos_enc)
        return data

    def concealer2sgat(self, data: Data, non_edge: bool, bidirection=None):
        r'''pyg_homograph->dgl_heterograph
            NOTE DGL internally decides a deterministic order for the same set of node types and canonical edge types, 
            which does not necessarily follow the order in data_dict.
        '''
        # bidirection relations
        if bidirection is None:
            bidirection = self.sgat_bidirection
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
            src_id, dst_id = edge[0], edge[1]
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

    def concealer_nid_mapping(self, g: DGLHeteroGraph, concealer_masks: list, id_map: dict):
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
            mask = edges_type.nonzero().squeeze(-1)
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

    def infer(self, doc, word2vec=None, pos_enc=None, dims=30):
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
        r'''Flash node embedding
        '''
        if word2vec is None:
            word2vec = self.flash_word2vec
        if pos_enc is None:
            pos_enc = self.flash_pos_enc
        data.y = data.x.clone().view(-1).long()
        docs = self.get_docs(data)
        x = [self.infer(doc, word2vec, pos_enc) for doc in docs]
        data.x = torch.from_numpy(np.array(x)).to(torch.float32)
        assert data.x.shape[0] == data.y.shape[0]
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

    def processed2wordvec(self, G: DiGraph):
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
    supply_preprocess()
