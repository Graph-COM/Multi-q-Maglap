import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List, Optional
import mmcv
import torch
from tqdm import tqdm
import numpy as np
import scipy.io as scio
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

class random_graph_distance(InMemoryDataset):
    def __init__(self, url=None, dataname='distance', root='data', processed_suffix='', split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.url = url
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataname)
        self.split = split
        self.processed_suffix = processed_suffix
        super(random_graph_distance, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                                 pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw/'+self.split
        return os.path.join(self.raw, name)

    @property
    def processed_dir(self):
        return os.path.join(self.raw, 'processed'+self.processed_suffix)

    @property
    def raw_file_names(self):
        names = os.listdir(self.raw_dir)
        return names

    @property
    def processed_file_names(self):
        return ['data_'+self.split+'.pt']

    def np2pyg(self, edge_index, dist):
        num_nodes = np.max(edge_index) + 1
        if dist.ndim == 2:
            dist_index = np.where((dist != np.inf) * (dist != 0))
            y = dist[dist_index] # link prediction labels
        elif dist.ndim == 3:
            dist_index = np.where(dist.sum(-1) > 0) # link with non-zero walk profile
            y = dist[dist_index]
        dist_index = np.concatenate((dist_index[0].reshape(1, -1), dist_index[1].reshape(1, -1)), axis=0)
        return Data(edge_index=torch.tensor(edge_index), num_nodes=num_nodes, dist_index=torch.tensor(dist_index),
                    y=torch.tensor(y).float())



    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        data_list = []
        count = 0
        for file in self.raw_paths:
            raw_data = np.load(file, allow_pickle=True)
            raw_data = raw_data['data']
            for data in raw_data:
                if data[0].shape[-1] == 0: # no edge graph
                    continue
                data_list.append(self.np2pyg(data[0], data[1]))
                count += 1
                if count % 10000 == 0:
                    print('Loading raw data: #%d' % count)
            #break
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            print('pre-transforming for data at ' + self.processed_paths[0])
            temp = []
            for i, data in enumerate(data_list):
                if i != 0 and i % 5000 == 0:
                    print('Pre-processing %d/%d' % (i, len(data_list)))
                    #break
                temp.append(self.pre_transform(data))
            data_list = temp
            # data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
