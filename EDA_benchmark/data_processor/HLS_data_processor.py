from tqdm import tqdm
import os
import random
import pickle
import numpy as np
from easydict import EasyDict
import os.path as osp
import pandas as pd
import re

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.transforms import Compose
import torch_geometric.transforms as T

from utils import create_nested_folder
from maglap.get_mag_lap import AddMagLaplacianEigenvectorPE, AddLaplacianEigenvectorPE


class HLSDataProcessor(InMemoryDataset):
    def __init__(self, config, mode):
        self.config = config
        self.save_folder = str(config['task']['processed_folder'])+str(config['task']['name'])+'/'+str(config['task']['type'])+'/'
        create_nested_folder(self.save_folder)
        self.divide_seed = config['task']['divide_seed']
        self.mode = mode
        self.raw_data_root = config['task']['raw_data_path']
        self.pe_type = config['model'].get('pe_type')
        if self.pe_type is None:
            pre_transform = None
        elif self.pe_type == 'lap':
            pre_transform = Compose([T.AddRandomWalkPE(walk_length = config['model']['se_pe_dim_input'], attr_name = 'rw_se')])
            self.lap_pre_transform = Compose([AddLaplacianEigenvectorPE(k=config['model']['lap_pe_dim_input'], attr_name='lap_pe')])
        elif self.pe_type == 'maglap':
            pre_transform = Compose([T.AddRandomWalkPE(walk_length = config['model']['se_pe_dim_input'], attr_name = 'rw_se')])
            self.mag_pre_transform = Compose([AddMagLaplacianEigenvectorPE(k=config['model']['mag_pe_dim_input'], q=config['model']['q'],
                                                         multiple_q=config['model']['q_dim'], attr_name='mag_pe')])
        super().__init__(root = self.save_folder, pre_transform = pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[mode])
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_dir(self) -> str:
        processed_dir = osp.join(self.save_folder, 'processed')
        if self.pe_type is None:
            processed_dir += '_no_pe'
        if self.pe_type == 'lap':
            processed_dir += '_' + self.pe_type + str(self.config['model']['lap_pe_dim_input'])
        elif self.pe_type == 'maglap':
            processed_dir += '_' + str(self.config['model']['mag_pe_dim_input']) + 'k_' + str(self.config['model']['q_dim']) + 'q' + str(self.config['model']['q'])
        return processed_dir
    
    @property
    def processed_file_names(self):
        return {
            'train': 'train_'+str(self.divide_seed)+'.pt',
            'valid': 'val_'+str(self.divide_seed)+'.pt',
            'test': 'test_'+str(self.divide_seed)+'.pt',
            'test_real': 'test_real_'+str(self.divide_seed)+'.pt',
            'test_othertype': 'test_other_type_'+str(self.divide_seed)+'.pt'
        }
    @property
    def processed_paths(self):
        return {mode: os.path.join(self.processed_dir, fname) for mode, fname in self.processed_file_names.items()}
    def process(self):
        file_names = self.processed_file_names
        # check if has already created
        exist_flag = 0
        for key in file_names:        
            exist_flag = exist_flag + os.path.isfile(self.processed_paths[key])
        if exist_flag == len(file_names):
            print('all datasets already exists, directly load.')
            return
        else:
            indices = list(range(18570))
            random.seed(123)
            random.shuffle(indices)
            indice_dict = {}
            train_indices = indices[:16570]
            valid_indices = indices[16570:17570]
            test_indices = indices[17570:]
            test_real_indices = list(range(18570, 18626))
            other_indices = list(range(19119))
            
            test_othertype_indices = random.sample(other_indices, 1000)
            indice_dict['train'] = train_indices
            indice_dict['valid'] = valid_indices
            indice_dict['test'] = test_indices
            indice_dict['test_real'] = test_real_indices
            indice_dict['test_othertype'] = test_othertype_indices
            cdfg_raw_data_path = self.config['task']['raw_data_path']+'cdfg'+'_cp_all/'
            dfg_raw_data_path = self.config['task']['raw_data_path']+'dfg'+'_cp/'
            cdfg_graph_list = self.read_csv_graph_raw(cdfg_raw_data_path, check_repeat_edge = False)
            dfg_graph_list = self.read_csv_graph_raw(dfg_raw_data_path, check_repeat_edge = True)
            for key in file_names:
                data_list = []
                if key != 'test_othertype':
                    for i, id in enumerate(indice_dict[key]):
                        if i % 1000 == 0:
                            print('pre-transforming ' + key + ' dataset: %d/%d' % (i, len(indice_dict[key])))
                        data = Data(x = torch.tensor(cdfg_graph_list[id]['node_feat']).long(), edge_index = torch.tensor(cdfg_graph_list[id]['edge_index']),
                                    edge_attr = torch.tensor(cdfg_graph_list[id]['edge_feat']).long(), dsp = torch.tensor(cdfg_graph_list[id]['dsp']).to(dtype=torch.float32),
                                    cp = torch.tensor(cdfg_graph_list[id]['cp']).to(dtype=torch.float32), lut = torch.tensor(cdfg_graph_list[id]['lut']).to(dtype=torch.float32),
                                    ff = torch.tensor(cdfg_graph_list[id]['ff']).to(dtype=torch.float32), slice = torch.tensor(cdfg_graph_list[id]['slice']).to(dtype=torch.float32))
                        # add undirected random walk SE
                        if self.pe_type in ['lap', 'maglap']:
                            if self.config['model']['se_pe_dim_input'] > 0:
                                bi_edge_index, bi_edge_weight = to_undirected(data.edge_index, data.edge_attr)
                                padded_data = self.add_padding(data, max(self.config['model'][self.pe_type[:3]+'_pe_dim_input'], self.config['model']['se_pe_dim_input']))
                                tmp_bidirect_data = Data(x = padded_data.x, edge_index = bi_edge_index, edge_attr = bi_edge_weight)
                                tmp_bidirect_data = self.pre_transform(tmp_bidirect_data)
                                data['rw_se'] = tmp_bidirect_data['rw_se']
                            if self.pe_type == 'lap':
                                lap_data = self.lap_pre_transform(data)
                                data['lap_pe'] = lap_data['lap_pe']
                                data['Lambda'] = lap_data['Lambda']
                            elif self.pe_type == 'maglap':
                                mag_data = self.mag_pre_transform(data)
                                data['mag_pe'] = mag_data['mag_pe']
                                data['Lambda'] = mag_data['Lambda']
                        data_list.append(data)
                else:  
                    for i, id in enumerate(indice_dict[key]):
                        if i % 1000 == 0:
                            print('pre-transforming ' + key + ' dataset: %d/%d' % (i, len(indice_dict[key])))
                        data = Data(x = torch.tensor(dfg_graph_list[id]['node_feat']).long(), edge_index = torch.tensor(dfg_graph_list[id]['edge_index']),
                                    edge_attr = torch.tensor(dfg_graph_list[id]['edge_feat']).long(), dsp = torch.tensor(dfg_graph_list[id]['dsp']).to(dtype=torch.float32),
                                    cp = torch.tensor(dfg_graph_list[id]['cp']).to(dtype=torch.float32), lut = torch.tensor(dfg_graph_list[id]['lut']).to(dtype=torch.float32),
                                    ff = torch.tensor(dfg_graph_list[id]['ff']).to(dtype=torch.float32), slice = torch.tensor(dfg_graph_list[id]['slice']).to(dtype=torch.float32))
                        # add random walk SE
                        if self.pe_type in ['lap', 'maglap']:
                            if self.config['model']['se_pe_dim_input'] > 0:
                                bi_edge_index, bi_edge_weight = to_undirected(data.edge_index, data.edge_attr)
                                padded_data = self.add_padding(data, max(self.config['model'][self.pe_type[:3]+'_pe_dim_input'], self.config['model']['se_pe_dim_input']))
                                tmp_bidirect_data = Data(x = padded_data.x, edge_index = bi_edge_index, edge_attr = bi_edge_weight)
                                tmp_bidirect_data = self.pre_transform(tmp_bidirect_data)
                                data['rw_se'] = tmp_bidirect_data['rw_se']
                            if self.pe_type == 'lap':
                                lap_data = self.lap_pre_transform(data)
                                data['Lambda'] = lap_data['Lambda']
                                data['lap_pe'] = lap_data['lap_pe']
                            elif self.pe_type == 'maglap':
                                mag_data = self.mag_pre_transform(data)
                                data['mag_pe'] = mag_data['mag_pe']
                                data['Lambda'] = mag_data['Lambda']
                        data_list.append(data)
                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[key])
    def read_csv_graph_raw(self, raw_dir, check_repeat_edge):
        label_dir = raw_dir + 'mapping'
        raw_dir = raw_dir + 'raw'
        labels = pd.read_csv(osp.join(label_dir, 'mapping.csv'))
        if isinstance(labels['DSP'][0], str):
            labels['DSP'] = labels['DSP'].apply(lambda x: float(re.findall(r'\d+\.?\d*', x)[0]) if x else None)
            labels['LUT'] = labels['LUT'].apply(lambda x: float(re.findall(r'\d+\.?\d*', x)[0]) if x else None).round(3)
            labels['CP'] = labels['CP'].apply(lambda x: float(re.findall(r'\d+\.?\d*', x)[0]) if x else None).round(3)
            labels['FF'] = labels['FF'].apply(lambda x: float(re.findall(r'\d+\.?\d*', x)[0]) if x else None).round(3)
            labels['SLICE'] = labels['SLICE'].apply(lambda x: float(re.findall(r'\d+\.?\d*', x)[0]) if x else None)
        try:
            edge = pd.read_csv(osp.join(raw_dir, 'edge.csv'), header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
            num_node_list = pd.read_csv(osp.join(raw_dir, 'num-node-list.csv'), header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
            num_edge_list = pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv'), header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list
        except FileNotFoundError:
            raise RuntimeError('No such file')
        try:
            node_feat = pd.read_csv(osp.join(raw_dir, 'node-feat.csv'), header = None).values
            if 'int' in str(node_feat.dtype):
                node_feat = node_feat.astype(np.int64)
            else:
                node_feat = node_feat.astype(np.float32)
        except FileNotFoundError:
            node_feat = None
        #[0 0 0 0 0 0 0]
        #[3 256 7 56 2 2 257]
        print('node feature min'+str(node_feat.min(axis = 0)))
        print('node feat max:'+str(node_feat.max(axis = 0)))
        #print(np.unique(node_feat[:, 6]))
        #print(np.unique(node_feat[:, 1]))

        edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv'), header = None).values
        if 'int' in str(edge_feat.dtype):
            edge_feat = edge_feat.astype(np.int64)
        else:
            edge_feat = edge_feat.astype(np.float32)

        print('edge feat min'+str(edge_feat.min(axis = 0)))
        print('edge feat max:'+str(edge_feat.max(axis = 0)))
        graph_list = []
        num_node_accum = 0
        num_edge_accum = 0
        print('Processing graphs...')
        for graph_id, (num_node, num_edge) in tqdm(enumerate(zip(num_node_list, num_edge_list))):
            graph = dict()
            graph['edge_index'] = edge[:, num_edge_accum:num_edge_accum+num_edge]
            graph['edge_feat'] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
            if check_repeat_edge:
                repeated_edge_index = self.check_repeat_edge(graph['edge_index'])
                indices_to_remove = [index[1] for index in repeated_edge_index if len(index) > 1]
                all_indices = set(range(graph['edge_index'].shape[1]))
                indices_to_keep = list(all_indices - set(indices_to_remove))
                graph['edge_index'] = graph['edge_index'][:,indices_to_keep]
                graph['edge_feat'] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
                graph['edge_feat'] = graph['edge_feat'][indices_to_keep]
                if graph['edge_index'].shape[1] != graph['edge_feat'].shape[0]:
                    import pdb; pdb.set_trace()
            num_edge_accum += num_edge
            ### handling node
            if node_feat is not None:
                graph['node_feat'] = node_feat[num_node_accum:num_node_accum+num_node]
            else:
                graph['node_feat'] = None
            # turn the node_feature into binary_encoding
            # original  7 dimension
            # min [0 0 0 0 0 0 0]
            # max [3 256 7 56 2 2 257]
            # now 2 + 8 + 3 + 6 + 2 + 2 + 8 = 31
            graph['dsp'] = labels['DSP'][graph_id]
            graph['lut'] = labels['LUT'][graph_id]
            graph['cp'] = labels['CP'][graph_id]
            graph['ff'] = labels['FF'][graph_id]
            graph['slice'] = labels['SLICE'][graph_id]
            graph['num_nodes'] = num_node
            num_node_accum += num_node
            graph_list.append(graph)
        return graph_list
    
    def check_repeat_edge(self, edges):
        normalized_edges = np.sort(edges, axis=0)
        edge_counts = {}
        for i in range(normalized_edges.shape[1]):
            edge = tuple(normalized_edges[:, i])
            if edge in edge_counts:
                edge_counts[edge].append(i)
            else:
                edge_counts[edge] = [i]
        repeated_edge_indices = [indices for indices in edge_counts.values() if len(indices) == 2]
        if len(repeated_edge_indices) > 0:
            print(len(repeated_edge_indices))
        return repeated_edge_indices

    
    def add_padding(self, data, target_size):
        num_nodes = data.num_nodes
        if num_nodes <= target_size:
            num_nodes_to_add = target_size - num_nodes + 1
            extra_node_features = torch.zeros((num_nodes_to_add, data.x.shape[1])).long()
            data.x = torch.cat([data.x, extra_node_features], dim=0)
        return data
    

    