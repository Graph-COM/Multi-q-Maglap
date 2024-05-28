import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import degree
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg
import pickle


def get_degree(data):
   data.degree = 1. / torch.sqrt(1 + degree(data.edge_index[0], data.num_nodes))
   return data

def symmetrize_transform(data):
    data.edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=-1)
    return data

def bidirect_transform(data):
    num_edges = data.edge_index.size(1)
    data.edge_attr = torch.cat([torch.zeros([num_edges]), torch.ones([num_edges])], dim=0).to(data.edge_index.device).int()
    data.edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=-1)
    return data

class SortingDataset(InMemoryDataset):
    def __init__(self, name, root='data', transform=None, pre_transform=None, pre_filter=None,
                 processed_suffix='', split='train'):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name  ## original name, e.g., ogbg-molhiv
        self.raw = os.path.join(root, name)
        self.processed_suffix = processed_suffix
        self.pre_filter = pre_filter
        self.split = split
        assert split in ['train', 'valid', 'test']


        super(SortingDataset, self).__init__(self.raw, transform, pre_transform, pre_filter=pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_dir(self):
        return os.path.join(self.raw, 'processed' + self.processed_suffix, self.split)

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        ### read pyg graph list
        data_path = osp.join(self.raw, self.split)
        files = os.listdir(data_path)
        full_data_list = []
        for file in files:
            if not file.endswith('pkl'):
                continue
            print('preprocessing %s' % file)
            with open(osp.join(data_path, file), 'rb') as f:
                data_list = pickle.load(f)

            # type
            data_list_new = []
            for data in data_list:
                data.x = data.x.int()
                data.edge_index = data.edge_index.to(torch.int64)
                data.seq_len = (data.x[:, 1:].max() + 1).view(1, 1) # mark the sequence length
                data_list_new.append(data)
            data_list = data_list_new

            if self.pre_filter is not None:
                #print('pre-filtering...')
                data_list_filter = []
                count = 0
                for i, data in enumerate(data_list):
                    if i % 5000 == 0:
                        print('pre-filtering: %d/%d' % (i, len(data_list)))
                    if self.pre_filter(data):
                        data_list_filter.append(data)
                        count += 1
                data_list = data_list_filter
                #data_list = [data for data in data_list if self.pre_filter(data)]
                #print('pre-filtering finished, num of data left %d' % len(data_list))


            if self.pre_transform is not None:
                #print('pre-transforming dataset...')
                data_list_new = []
                for i, data in enumerate(data_list):
                    if i % 5000 == 0:
                        print('pre-transforming: %d/%d' % (i, len(data_list)))
                    data_list_new.append(self.pre_transform(data))
                #data_list = [self.pre_transform(data) for data in data_list]
                data_list = data_list_new

            full_data_list += data_list


        data, slices = self.collate(full_data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


