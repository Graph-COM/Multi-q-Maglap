

import torch
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential, Dropout

from torch_geometric.nn import GINEConv, GCNConv, GPSConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.attention import PerformerAttention

from torch_geometric_signed_directed.nn import MSConv
from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer
from torch_geometric.utils import to_dense_batch

from models.base_model import BaseModel, MLPs
from models.pe_encoders import PEEmbedderWrapper
from maglap.handle_complex import ComplexHandler, SparseComplexNetwork, DenseComplexNetwork

class RedrawProjection:
    def __init__(self, model, redraw_interval):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0
    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class MiddleModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # parameters for middle model
        self.num_layers = args['num_layers']
        self.pe_type = args.get('pe_type')
        self.pe_strategy = args.get('pe_strategy')
        self.q_dim = args.get('q_dim')

        # parameters for base model
        self.base_model_name = args['name']
        self.dropout = args['dropout']
        self.hidden_dim = args['hidden_dim']
        self.inner_gnn = args.get('inner_gnn')
        
        
        # allocate base model convs
        self.convs = ModuleList()
        for i in range(self.num_layers):
            #if (self.pe_type is None) or (self.pe_strategy in ['invariant_fixed', 'invariant_learnable']):
            if args.get('pe_embedder') is None:
                # the case where node features are from raw node attributes
                conv = BaseModel(base_model = self.base_model_name, hidden_dim = self.hidden_dim, dropout = self.dropout, inner_gnn = self.inner_gnn)
            #elif args.get('pe_embedder') is not None:
            else:
                # the case where node features are from both raw node attributes and pe embedder
                self.pe_output_dim = args.get(self.pe_type[:3]+'_pe_dim_output')
                conv = BaseModel(base_model = self.base_model_name, hidden_dim = self.hidden_dim+self.pe_output_dim, dropout = self.dropout, inner_gnn = self.inner_gnn)
            self.convs.append(conv)
        if self.base_model_name in ['GPS','GPSSE']:
            self.redraw_projection = RedrawProjection(self.convs, redraw_interval=None)
        
        # for pe
        if self.pe_type is not None:
            self.pe_strategy = args['pe_strategy']
            #pe_dim = 2 * self.q_dim * int(args['mag_pe_dim_input']) if self.pe_type == 'maglap' else int(args['lap_pe_dim_input'])
            pe_dim = args.get('mag_pe_dim_input') if self.pe_type == 'maglap' else args.get('lap_pe_dim_input')
            if self.pe_strategy == 'variant':
                #eigval_dim = pe_dim // 2 if self.pe_type == 'maglap' else pe_dim
                #pe_dim_output = args[self.pe_type[:3]+'_pe_dim_output']
                #self.pe_projection = Linear(pe_dim + eigval_dim, pe_dim_output)
                self.pe_projection = PEEmbedderWrapper(args)
            elif self.pe_strategy == 'invariant_fixed':
                q_dim = self.q_dim if self.pe_type == 'maglap' else 1
                # if use a node-level PE before GNN
                if args.get('pe_embedder') is not None:
                    self.pe_projection = PEEmbedderWrapper(args)
                else:
                    self.pe_projection = None
                # for processing invariant/equivariant pe
                desired_pe_edge_dim = args['hidden_dim']
                if args.get('pe_embedder') is not None:
                    desired_pe_edge_dim += self.pe_output_dim
                self.sparse_pe_net = SparseComplexNetwork(pe_dim, q_dim, self.pe_type, desired_pe_edge_dim,
                                                          network_type=args['pe_encoder'])
                if self.base_model_name in ['GPS']: # or other transformer based model?
                   self.dense_pe_net = DenseComplexNetwork(pe_dim, q_dim, self.pe_type, 4,
                                                           network_type=args['pe_encoder']) # 4 is # heads of multi-head attention
                #if self.pe_type == 'maglap':
                #self.complex_handler = ComplexHandler(pe_dim=pe_dim, q_dim=q_dim, pe_type=self.pe_type)
                #self.eigval_encoder = MLPs(args['eigval_encoder']['in'], args['eigval_encoder']['hidden'],
                                        # args['eigval_encoder']['out'], args['eigval_encoder']['num_layer'])
                #self.attn_bias_projection = Linear(2 * q_dim * 8, 4)
            elif self.pe_strategy == 'invariant_learnable':
                print('not implemented yet!')
    @property
    def out_dims(self) -> int:
        if self.pe_type is None:
            return self.hidden_dim
        #elif self.pe_strategy == 'variant':
        elif self.pe_projection is not None:
            return self.hidden_dim + self.pe_output_dim
        else:
            return self.hidden_dim
        #elif self.pe_strategy == 'invariant_fixed':
        #    return self.hidden_dim
        #elif self.pe_strategy == 'invariant_learnable':
        #    print('not implemented yet!')
    
    def forward(self, x, edge_index, batch, **kwargs):
        # kwargs:
        if self.pe_type is None:
            return self.no_pe_forward(x, edge_index, batch, **kwargs)
        elif self.pe_strategy == 'variant':
            return self.variant_forward(x, edge_index, batch, **kwargs)
        elif self.pe_strategy == 'invariant_fixed':
            return self.invariant_fixed_forward(x, edge_index, batch, **kwargs)
        elif self.pe_strategy == 'invariant_learnable':
            print('not implemented yet')
    def no_pe_forward(self, x, edge_index, batch, **kwargs):
        for conv in self.convs:
            x = conv(x, edge_index, batch, **kwargs)
        return x
    def variant_forward(self, x, edge_index, batch, **kwargs):
        z = kwargs[self.pe_type[:3]+'_pe']
        #z = torch.cat([z, kwargs['Lambda'][batch]], dim=-1) # concat eigenvalues
        z = self.pe_projection(z, kwargs['Lambda'], edge_index, batch)
        x = torch.cat([x, z], dim=-1)
        for conv in self.convs:
            x = conv(x = x, edge_index = edge_index, batch = batch, **kwargs)
        return x
    def invariant_fixed_forward(self, x, edge_index, batch, **kwargs):
        z = kwargs[self.pe_type[:3]+'_pe']

        if self.pe_projection is not None:
            node_pe = self.pe_projection(z, kwargs['Lambda'], edge_index, batch)
            x = torch.cat([x, node_pe], dim=-1)

        if 'edge_attr' in kwargs:
            edge_attr = kwargs['edge_attr'] + self.sparse_pe_net(z, kwargs['Lambda'], edge_index, batch)
        else:
            edge_attr = self.sparse_pe_net(z, kwargs['Lambda'], edge_index, batch)
        #import pdb; pdb.set_trace()

        if self.base_model_name == 'GPS':
            #pe, _ = to_dense_batch(batch[self.args['pe_type'][:3]+'_pe'], batch)
            #Lambda = kwargs['Lambda'].unflatten(-1, (self.complex_handler.q_dim, -1)) # [B, Q, pe_dim]
            #Lambda = self.eigval_encoder(Lambda.unsqueeze(-1))  # [B, Q, pe_dim, 8]
            #attn_bias = self.complex_handler.weighted_gram_matrix_batched(pe, Lambda) # [B, N, N, q_dim * 8]
            #attn_bias = self.attn_bias_projection(attn_bias) # [B, N, N, #heads]
            #attn_bias = torch.transpose(attn_bias, 1, -1).flatten(0, 1) # [B*#heads, N, N]
            attn_bias = self.dense_pe_net(kwargs[self.pe_type[:3]+'_pe'], kwargs['Lambda'], batch)
            attn_bias = torch.transpose(attn_bias, 1, -1).flatten(0, 1)  # [B*#heads, N, N]
            for conv in self.convs:
                # move computation of edge_attr/attn_bias to here to get a different edge_attr/attn_bias for each layer
                x = conv(x, edge_index, batch, edge_attr = edge_attr, attn_bias=attn_bias)
        else:
            for conv in self.convs:
                # move computation of edge_attr/attn_bias to here to get a different edge_attr/attn_bias for each layer
                x = conv(x, edge_index, batch, edge_attr = edge_attr)
        return x