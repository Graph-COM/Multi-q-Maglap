

import torch
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential, Dropout

from torch_geometric.nn import GINEConv, GCNConv, GINConv, GATConv, GPSConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.attention import PerformerAttention

from torch_geometric_signed_directed.nn import MSConv
from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer

from models.gps_conv import GPSConv_custom
#from models.mamba.mamba import MAMBAConv


class BaseModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # kwargs: hidden_dim, base_model, dropout
        # if GPS: inner_gnn
        self.hidden_dim = kwargs['hidden_dim']
        self.base_model = kwargs['base_model']
        self.dropout = kwargs['dropout']
        if self.base_model in ['GINE', 'DIGINE']:
            nn = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout),)
            self.conv = GINEConv(nn)
        elif self.base_model in ['GIN', 'DIGIN']:
            nn = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout),)
            self.conv = GINConv(nn)
        elif self.base_model in ['GCN', 'DIGCN']:
            self.conv = GCNConv(self.hidden_dim, self.hidden_dim)
        elif self.base_model == 'BIGINE':
            nn_forward = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout))
            self.conv_forward = GINEConv(nn_forward)
            nn_backward = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout))
            self.conv_backward = GINEConv(nn_backward)
        elif self.base_model == 'BIGIN':
            nn_forward = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout))
            self.conv_forward = GINConv(nn_forward)
            nn_backward = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout))
            self.conv_backward = GINConv(nn_backward)
        elif self.base_model == 'BIGCN':
            self.conv_forward = GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv_backward = GCNConv(self.hidden_dim, self.hidden_dim)
        elif self.base_model == 'MSGNN':
            self.complex_relu = complex_relu_layer()
            self.conv = MSConv(self.hidden_dim, self.hidden_dim//2, K = 2, q = 0.25, trainable_q = True)
        elif self.base_model == 'GPS':
            if kwargs['inner_gnn'] == 'GINE': 
                nn = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout))
                self.conv = GPSConv_custom(self.hidden_dim, 
                                GINEConv(nn), 
                                heads=4, attn_type='multihead', attn_kwargs = {'dropout': self.dropout})
            elif kwargs['inner_gnn'] == 'GIN':
                nn = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout))
                self.conv = GPSConv_custom(self.hidden_dim, 
                                GINConv(nn), 
                                heads=4, attn_type='multihead', attn_kwargs = {'dropout': self.dropout})
        elif self.base_model == 'PERFORMER':
            if kwargs['inner_gnn'] == 'GINE': 
                nn = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout))
                self.conv = GPSConv(self.hidden_dim, 
                                GINEConv(nn), attn_type='performer', attn_kwargs = {'dropout': self.dropout})
            elif kwargs['inner_gnn'] == 'GIN':
                nn = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(self.dropout))
                self.conv = GPSConv(self.hidden_dim, 
                                GINConv(nn), attn_type='performer', attn_kwargs = {'dropout': self.dropout})
        elif self.base_model in ['GAT', 'DIGAT']:
            self.conv = GATConv(int(self.hidden_dim), int(self.hidden_dim))
        elif self.base_model == 'BIGAT':
            self.conv_forward = GATConv(int(self.hidden_dim), int(self.hidden_dim))
            self.conv_backward = GATConv(int(self.hidden_dim), int(self.hidden_dim))
        elif self.base_model == 'MAMBA':
            assert kwargs.get('inner_gnn') in ['GIN', 'GINE']
            gin_conv = GINEConv if kwargs['inner_gnn'] == 'GINE' else GINConv
            nn = Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Dropout(self.dropout))
            
            self.conv = GPSConv_custom(self.hidden_dim,
                            gin_conv(nn),
                            heads=4, attn_type='multihead', attn_kwargs = {'dropout': self.dropout})

    def forward(self, x, edge_index, batch, **kwargs):
        if self.base_model in ['GINE', 'DIGINE']:
            x = self.conv(x, edge_index, kwargs['edge_attr'])
        elif self.base_model in ['GIN', 'DIGIN', 'GCN']:
            x = self.conv(x, edge_index)
        elif self.base_model == 'BIGINE':
            reverse_edge_index = edge_index[[1, 0], :]
            x1 = self.conv_forward(x, edge_index, kwargs['edge_attr'])
            x2 = self.conv_backward(x, reverse_edge_index, kwargs['edge_attr'])
            x = (x1 + x2)/2
        elif self.base_model in ['BIGIN', 'BIGCN']:
            reverse_edge_index = edge_index[[1, 0], :]
            x1 = self.conv_forward(x, edge_index)
            x2 = self.conv_backward(x, reverse_edge_index)
            x = (x1 + x2)/2
        elif self.base_model in ['GAT', 'DIGAT']:
            if kwargs.get('edge_attr') is None:
                x = self.conv(x, edge_index)
            else:
                x = self.conv(x, edge_index, edge_attr = kwargs.get('edge_attr'))
        elif self.base_model == 'BIGAT':
            reverse_edge_index = edge_index[[1, 0], :]
            if kwargs.get('edge_attr') is None:
                x1 = self.conv_forward(x, edge_index)
                x2 = self.conv_backward(x, reverse_edge_index)
            else:
                x1 = self.conv_forward(x, edge_index, edge_attr = kwargs.get('edge_attr'))
                x2 = self.conv_backward(x, reverse_edge_index, edge_attr = kwargs.get('edge_attr'))
            x = (x1 + x2)/2
        elif self.base_model == 'MSGNN':
            x_imag = x.clone()
            if kwargs.get('edge_attr') is not None:
                edge_attr = torch.sum(kwargs['edge_attr'], dim = 1)
                x, x_imag = self.conv(x, x_imag, edge_index, edge_attr)
            else:
                x, x_imag = self.conv(x, x_imag, edge_index)
            x, x_imag = self.complex_relu(x, x_imag)
            x = torch.cat((x, x_imag), dim = -1)
        elif self.base_model == 'GPS':
            if kwargs.get('edge_attr') is None:
                x = self.conv(x = x, edge_index = edge_index, batch = batch, attn_bias = kwargs.get('attn_bias'))
            else:
                x = self.conv(x = x, edge_index = edge_index, batch = batch, edge_attr = kwargs.get('edge_attr'), attn_bias = kwargs.get('attn_bias'))
        elif self.base_model == 'MAMBA':
            if kwargs.get('edge_attr') is None:
                x = self.conv(x = x, edge_index = edge_index, batch = batch)
            else:
                x = self.conv(x = x, edge_index = edge_index, batch = batch, edge_attr = kwargs.get('edge_attr'))
        elif self.base_model == 'PERFORMER':
            if kwargs.get('edge_attr') is None:
                x = self.conv(x = x, edge_index = edge_index, batch = batch)
            else:
                x = self.conv(x = x, edge_index = edge_index, batch = batch, edge_attr = kwargs.get('edge_attr'))
        return x
    

from torch import nn

class MLPs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, norm=None):
        super(MLPs, self).__init__()
        assert num_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        if norm == 'bn':
            self.layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm == 'ln':
            self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm == 'bn':
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm == 'ln':
                self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)