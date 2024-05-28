from types import CellType
from tqdm import tqdm
import os
import random
import pickle
import numpy as np
from easydict import EasyDict
import os.path as osp
import pandas as pd
import re
from itertools import combinations

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.transforms import Compose
import torch_geometric.transforms as T

from utils import create_nested_folder
from maglap.get_mag_lap import AddMagLaplacianEigenvectorPE, AddLaplacianEigenvectorPE


class AMPDataProcessor(InMemoryDataset):
    def __init__(self, config, mode):
        self.config = config
        self.save_folder = str(config['task']['processed_folder'])+str(config['task']['name'])+'/'
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
            if config['model'].get('dynamic_q') == 1:
                self.dynamic_q = True
            else:
                self.dynamic_q = False
            pre_transform = Compose([T.AddRandomWalkPE(walk_length = config['model']['se_pe_dim_input'], attr_name = 'rw_se')])
            self.mag_pre_transform = Compose([AddMagLaplacianEigenvectorPE(k=config['model']['mag_pe_dim_input'], q=config['model']['q'],
                                                         multiple_q=config['model']['q_dim'], attr_name='mag_pe',
                                                                           dynamic_q=self.dynamic_q)])
        super().__init__(root = self.save_folder, pre_transform = pre_transform)
        if mode == 'train':
            self.data, self.slices = torch.load(self.processed_paths['train'])
        elif mode == 'valid':
            self.data, self.slices = torch.load(self.processed_paths['valid'])
        elif mode == 'test_id':
            self.data, self.slices = torch.load(self.processed_paths['test'])
        elif mode == 'test_ood':
            self.data, self.slices = torch.load(self.processed_paths['ood'])
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
            if self.dynamic_q:
                processed_dir += '_dynamic'
        return processed_dir
    
    @property
    def processed_file_names(self):
        return {
            'train': 'train.pt',
            'valid': 'valid.pt',
            'test': 'test.pt',
            'ood': 'ood.pt',
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
            
            raw_data_path = self.config['task']['raw_data_path']
            graph_list, stage_2_indices, stage_3_indices = self.read_csv_graph_raw(raw_data_path)
            np.random.seed(123)
            np.random.shuffle(stage_3_indices)
            np.random.shuffle(stage_2_indices)
            num_training = int(len(stage_3_indices) * 0.9)
            num_validation = int(len(stage_3_indices) * 0.05)
            num_test = int(len(stage_3_indices) * 0.05)
            '''num_training = int(10000 * 0.9)
            num_validation = int(10000 * 0.05)
            num_test = int(10000 * 0.05)'''
            num_test_ood = 500

            '''indices__ = np.arange(10000)
            train_indices = indices__[num_test + num_validation:]
            valid_indices = indices__[num_test:num_test + num_validation]
            test_indices = indices__[:num_test]'''
            train_indices = stage_3_indices[:num_training]
            valid_indices = stage_3_indices[num_training:num_training + num_validation]
            test_indices = stage_3_indices[num_training + num_validation:]
            test_ood_indices = stage_2_indices[:num_test_ood]
            
            train_data_list = []
            val_data_list = []
            test_data_list = []
            test_ood_data_list = []
            
            for id, graph in tqdm(enumerate(graph_list)):
                data = Data(x = torch.tensor(graph_list[id]['node_feat']).to(dtype=torch.long), 
                            edge_index = torch.tensor(graph_list[id]['all_edge_index']).to(dtype=torch.long), 
                            sub_edge_index = torch.tensor(graph_list[id]['sub_edge_index']).to(dtype=torch.long),
                            gain = torch.tensor(graph_list[id]['gain']).reshape(-1, 1).to(dtype=torch.float32),
                            pm = torch.tensor(graph_list[id]['pm']).reshape(-1, 1).to(dtype=torch.float32),
                            bw = torch.tensor(graph_list[id]['bw']).reshape(-1, 1).to(dtype=torch.float32))
                full_edge_index = data.edge_index
                # add undirected random walk SE
                if self.pe_type is not None:
                    if self.config['model']['se_pe_dim_input'] > 0:
                        bi_edge_index = to_undirected(full_edge_index)
                        tmp_bidirect_data = Data(x = data.x, edge_index = bi_edge_index)
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
                # append to a list
                stage = graph_list[id]['stage']
                if stage == 3:
                    if id in train_indices:
                        train_data_list.append(data)
                    elif id in valid_indices:
                        val_data_list.append(data)
                    elif id in test_indices:
                        test_data_list.append(data)
                elif stage == 2:
                    if id in test_ood_indices:
                        test_ood_data_list.append(data)

            train_data, train_slices = self.collate(train_data_list)
            torch.save((train_data, train_slices), self.processed_paths['train'])

            valid_data, valid_slices = self.collate(val_data_list)
            torch.save((valid_data, valid_slices), self.processed_paths['valid'])

            test_data, test_slices = self.collate(test_data_list)
            torch.save((test_data, test_slices), self.processed_paths['test'])

            test_ood_data, test_ood_slices = self.collate(test_ood_data_list)
            torch.save((test_ood_data, test_ood_slices), self.processed_paths['ood'])

    def read_csv_graph_raw(self, raw_dir):
        label_path = osp.join(raw_dir, 'perform101.csv')
        edge_path = osp.join(raw_dir, 'edge.csv')
        node_path = osp.join(raw_dir, 'node-feat.csv')
        num_node_path = osp.join(raw_dir, 'num-node-list.csv')
        num_edge_path = osp.join(raw_dir, 'num-edge-list.csv')
        stage_path = osp.join(raw_dir, 'amp-stage.csv')
        
        node_feat = pd.read_csv(node_path, header = None).values
        all_edge_index = pd.read_csv(edge_path, header = None).values.T.astype(np.int64)
        labels = pd.read_csv(label_path, header = None).values.T # 6 * 10000 [nan, valid, gain, pm, bw, foc]
        labels = labels[:,1:]
        stages = pd.read_csv(stage_path, header = None).values.T

        statistics = {}
        statistics['gain_mean'] = np.mean(labels[2,:].astype(float))
        statistics['gain_std'] = np.std(labels[2,:].astype(float))
        statistics['pm_mean'] = np.mean(labels[3,:].astype(float))
        statistics['pm_std'] = np.std(labels[3,:].astype(float))
        statistics['bw_mean'] = np.mean(labels[4,:].astype(float))
        statistics['bw_std'] = np.std(labels[4,:].astype(float))
        target_path = self.save_folder + 'stats.pkl'
        with open(target_path, 'wb') as file:
            pickle.dump(statistics, file)
        
        num_node_list = pd.read_csv(num_node_path, header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
        num_edge_list = pd.read_csv(num_edge_path, header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list

        graph_list = []
        num_node_accum = 0
        num_edge_accum = 0
        stage_2_indices = []
        stage_3_indices = []
        for graph_id, (num_node, num_edge) in tqdm(enumerate(zip(num_node_list, num_edge_list))):
            graph = dict()
            graph['all_edge_index'] = all_edge_index[:, num_edge_accum:num_edge_accum+num_edge]
            num_edge_accum += num_edge
            graph['node_feat'] = node_feat[num_node_accum:num_node_accum+num_node]
            sub_edge_index = self.get_subgraph_edge(graph['node_feat'][:,1])
            graph['sub_edge_index'] = sub_edge_index
            graph['stage'] = stages[0][graph_id]
            if graph['stage'] == 2:
                stage_2_indices.append(graph_id)
            elif graph['stage'] == 3:
                stage_3_indices.append(graph_id)
            graph['gain'] = float(labels[2][graph_id])
            graph['pm'] = float(labels[3][graph_id])
            graph['bw'] = float(labels[4][graph_id])
            graph['num_nodes'] = num_node
            num_node_accum += num_node
            graph_list.append(graph)
            
        return graph_list, stage_2_indices, stage_3_indices
    
    def get_subgraph_edge(self, node_feat):
        index_groups = {}
        for i, value in enumerate(node_feat):
            if value not in index_groups:
                index_groups[value] = []
            index_groups[value].append(i)
        index_groups_list = list(index_groups.values())
        edges = []
        for group_id, group in enumerate(index_groups_list):
            if len(group) > 1:
                value_combinations = list(combinations(group, 2))
                directed_edges = [(min(a, b), max(a, b)) for a, b in value_combinations]
                directed_edges = np.array(directed_edges).T 
                edges.append(directed_edges)
        if len(edges) != 0:
            edges = np.hstack(edges)
        return edges
    