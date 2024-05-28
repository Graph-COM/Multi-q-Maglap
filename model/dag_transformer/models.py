# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch_geometric.nn as gnn
from .layers import TransformerEncoderLayer
from einops import repeat
import math
from torch_geometric.nn import GINEConv
import torch.nn.functional as F
from utils.handle_complex import SparseComplexNetwork
from torch_geometric.utils import *


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, SAT, edge_index, mask_dag_, dag_rr_edge_index,
            edge_attr=None, ptr=None, return_attn=False, dag_rr_edge_attr=None):
        output = x
        for mod in self.layers:
            output = mod(output, SAT, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=edge_attr,
                ptr=ptr,
                return_attn=return_attn,
                dag_rr_edge_attr=dag_rr_edge_attr,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False,
                 gnn_type="gcn",
                 use_global_pool=True, max_seq_len=None,
                 global_pool='mean', SAT=False, **kwargs):
        super().__init__()

        self.SAT = SAT
        self.dropout = nn.Dropout(0.1)
        self.gnn_type = gnn_type
        self.hidden_dim = d_model
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool



    #def forward(self, data, return_attn=False):
    def forward(self, x, edge_index, edge_attr, batch, dag_rr_edge_index, ptr, return_attn=False):
        #x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        #node_depth = data.node_depth if hasattr(data, "node_depth") else None
        #mask_dag_ = data.mask_rc if hasattr(data, 'mask_rc') else None
        mask_dag_ = None
        #dag_rr_edge_index = data.dag_rr_edge_index if hasattr(data, 'dag_rr_edge_index') else None
        #output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
        output = self.dropout(x)

        #if self.use_edge_attr and edge_attr is not None:
        #    edge_attr = self.embedding_edge(edge_attr)
        #else:
        #    edge_attr = None

        output = self.encoder(
            output, 
            self.SAT,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=edge_attr,
            ptr=ptr,
            return_attn=return_attn
        )
        return output
        # readout step
        #if self.use_global_pool:
        #    output = self.pooling(output, data.batch)

        #if self.max_seq_len is not None:
        #    pred_list = []
        #    for i in range(self.max_seq_len):
        #        pred_list.append(self.classifier[i](output))
        #    return pred_list
        #return self.classifier(output)

    @property
    def out_dims(self):
        return self.hidden_dim



class GraphTransformerInvariant(nn.Module):
    def __init__(self, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False,
                 gnn_type="gcn",
                 use_global_pool=True, max_seq_len=None,
                 global_pool='mean', SAT=False, **kwargs):
        super().__init__()

        self.SAT = SAT
        self.dropout = nn.Dropout(0.1)
        self.gnn_type = gnn_type
        self.hidden_dim = d_model
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        pe_dim = kwargs['pe_dim']
        q_dim = kwargs['q_dim']
        pe_type = kwargs['pe']
        complex_net_type = kwargs['handle_symmetry']
        self.sparse_pe_encoder = SparseComplexNetwork(pe_dim, q_dim, pe_type, d_model,
                                                           network_type=complex_net_type)


    #def forward(self, data, return_attn=False):
    def forward(self, x, pe, Lambda, edge_index, edge_attr, batch, dag_rr_edge_index, ptr, return_attn=False):
        #x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        #node_depth = data.node_depth if hasattr(data, "node_depth") else None
        #mask_dag_ = data.mask_rc if hasattr(data, 'mask_rc') else None
        mask_dag_ = None
        #dag_rr_edge_index = data.dag_rr_edge_index if hasattr(data, 'dag_rr_edge_index') else None
        #output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
        output = self.dropout(x)

        #if self.use_edge_attr and edge_attr is not None:
        #    edge_attr = self.embedding_edge(edge_attr)
        #else:
        #    edge_attr = None

        # compute pe-based edge attr for dag_rr_edges
        edge_attr_pe = self.sparse_pe_encoder(pe, Lambda, dag_rr_edge_index, batch)

        output = self.encoder(
            output,
            self.SAT,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=edge_attr,
            ptr=ptr,
            return_attn=return_attn,
            dag_rr_edge_attr=edge_attr_pe,
        )
        return output
        # readout step
        #if self.use_global_pool:
        #    output = self.pooling(output, data.batch)

        #if self.max_seq_len is not None:
        #    pred_list = []
        #    for i in range(self.max_seq_len):
        #        pred_list.append(self.classifier[i](output))
        #    return pred_list
        #return self.classifier(output)

    @property
    def out_dims(self):
        return self.hidden_dim


class GINE(torch.nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False,
                 gnn_type="gcn", use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', SAT=False, **kwargs):
        super().__init__()
        hidden_dim = d_model
        self.hidden_dim = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.edge_encoders = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                #nn.BatchNorm1d(self.hidden_dim)
                )
            conv = GINEConv(mlp, edge_dim=2)
            self.convs.append(conv)
            self.edge_encoders.append(torch.nn.Linear(128, d_model))
            self.norms.append(torch.nn.BatchNorm1d(d_model))
            #self.norms.append(InstanceNorm(self.hidden_dim))

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                                                out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1, ))

        for i, conv in enumerate(self.convs):
            output = conv(output, edge_index, edge_attr = edge_attr)
            output = self.norms[i](output)
            if i != len(self.convs) - 1:
                output = F.relu(output)
            #x = self.norms[i](x, batch)
        output = gnn.global_mean_pool(output, data.batch)
        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list
        return self.classifier(output)

