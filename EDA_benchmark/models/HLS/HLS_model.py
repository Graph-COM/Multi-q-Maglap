 


import torch
from torch import nn
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential, Dropout

from torch_geometric.nn import GINEConv, GCNConv, GPSConv
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.nn.attention import PerformerAttention

#from torch_geometric_signed_directed.nn import MSConv
#from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer

from models.base_model import MLPs
from models.middle_model import MiddleModel


class NodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_embedding_list = torch.nn.ModuleList()
        feature_dim_list = [4, 257, 8, 57, 3, 3, 258]
        for dim in feature_dim_list:
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.node_embedding_list.append(emb)
    def forward(self, x):
        x_embedding_list = []
        for i in range(x.shape[1]):
            x_embedding_list.append(self.node_embedding_list[i](x[:,i]))
        x_embedding = torch.cat(x_embedding_list, dim = 1)
        return x_embedding

class EdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(EdgeEncoder, self).__init__()
        self.edge_embedding_list = torch.nn.ModuleList()
        feature_dim_list = [4, 4]
        for dim in feature_dim_list:
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.edge_embedding_list.append(emb)
    def forward(self, edge_attr):
        edge_embedding_list = []
        for i in range(edge_attr.shape[1]):
            edge_embedding_list.append(self.edge_embedding_list[i](edge_attr[:,i]))
        edge_embedding = torch.cat(edge_embedding_list, dim = 1)
        return edge_embedding   


class HLSModel(nn.Module):
    def __init__(self, args, target):
        super(HLSModel, self).__init__()
        self.target = target
        self.args = args
        self.pe_type = args.get('pe_type')
        # define node, edge encoder
        node_emb_dim = args['hidden_dim'] // 7
        #if self.pe_type is not None and args['pe_strategy'] == 'variant':
        if self.pe_type is not None and args.get('pe_embedder') is not None:
            pe_dim_output = args['mag_pe_dim_output'] if self.pe_type == 'maglap' else args['lap_pe_dim_output']
            edge_emb_dim = (args['hidden_dim']+pe_dim_output) // 2
        else:
            edge_emb_dim = (args['hidden_dim']) // 2
        self.node_encoder = NodeEncoder(node_emb_dim)
        self.edge_encoder = EdgeEncoder(edge_emb_dim)

        # define middle model
        self.middle_model = MiddleModel(self.args)

        # define final layer
        self.output_mlp = MLPs(self.middle_model.out_dims, self.middle_model.out_dims, 1, args['mlp_out']['num_layer'])

    def forward(self, batch_data):
        #pre-process the node, edge data
        x = self.node_encoder(batch_data.x)
        edge_attr = self.edge_encoder(batch_data.edge_attr)
        #call the middle model to process the data
        if self.pe_type is None:
            x = self.middle_model(x, batch_data.edge_index, batch_data.batch, edge_attr = edge_attr)
        else:
            x = self.middle_model(x, batch_data.edge_index, batch_data.batch, edge_attr = edge_attr, 
                                  mag_pe = getattr(batch_data, 'mag_pe', None), lap_pe = getattr(batch_data, 'lap_pe', None),
                                  Lambda = batch_data.Lambda)
        # global pool and final MLP
        if self.target == 'cp':
            x = global_max_pool(x, batch_data.batch)
        elif self.target in ['dsp', 'lut']:
            x = global_add_pool(x, batch_data.batch)
        x = self.output_mlp(x)
        return x
        















class GINE(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args['hidden_dim'])
        self.node_emb = NodeEncoder(self.hidden_dim//7)
        self.edge_emb = EdgeEncoder(self.hidden_dim//2)
        self.convs = ModuleList()
        for _ in range(int(args['num_layers'])):
            nn = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(args['dropout']),
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(args['dropout']),)
            conv = GINEConv(nn)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            BatchNorm1d(self.hidden_dim // 2),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            BatchNorm1d(self.hidden_dim // 4),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)
    

class GCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args['hidden_dim'])
        self.node_emb = NodeEncoder(self.hidden_dim//7)
        self.convs = ModuleList()
        for _ in range(int(args['num_layers'])):
            conv = GCNConv(self.hidden_dim, self.hidden_dim)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            BatchNorm1d(self.hidden_dim // 2),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            BatchNorm1d(self.hidden_dim // 4),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_add_pool(x, batch)
        return self.mlp(x)
    

class NPNAS(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args['hidden_dim'])
        self.node_emb = NodeEncoder(self.hidden_dim//7)
        self.edge_emb = EdgeEncoder(self.hidden_dim//2)
        self.convs_forward = ModuleList()
        self.convs_backward = ModuleList()
        for _ in range(int(args['num_layers'])):
            nn_forward = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(args['dropout']),
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(args['dropout']),)
            nn_backward = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(args['dropout']),
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Dropout(args['dropout']),)
            conv_forward = GINEConv(nn_forward)
            conv_backward = GINEConv(nn_backward)
            self.convs_forward.append(conv_forward)
            self.convs_backward.append(conv_backward)

        self.mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            BatchNorm1d(self.hidden_dim // 2),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            BatchNorm1d(self.hidden_dim // 4),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        reverse_edge_index = edge_index[[1, 0], :]
        for conv_forward, conv_backward in zip(self.convs_forward, self.convs_backward):
            x_forward = conv_forward(x, edge_index, edge_attr)
            x_backward = conv_backward(x, reverse_edge_index, edge_attr)
            x = (x_forward + x_backward) / 2
        x = global_add_pool(x, batch)
        return self.mlp(x)
    
class MSGNN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args['hidden_dim'])
        self.node_emb = NodeEncoder(self.hidden_dim//7)
        #self.edge_emb = EdgeEncoder(self.hidden_dim//2)
        self.convs = ModuleList()
        self.complex_relu = complex_relu_layer()
        for _ in range(int(args['num_layers'])):
            conv = MSConv(self.hidden_dim, self.hidden_dim, K = 2, q = 0.25, trainable_q = True)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(self.hidden_dim * 2, self.hidden_dim // 2),
            BatchNorm1d(self.hidden_dim // 2),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            BatchNorm1d(self.hidden_dim // 4),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 4, 1),
        )
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        x_imag = x.clone()
        edge_attr = edge_attr[:, 0] + edge_attr[:, 1]
        for conv in self.convs:
            x, x_imag = conv(x, x_imag, edge_index, edge_attr)
            x, x_imag = self.complex_relu(x, x_imag)
        x = torch.cat((x, x_imag), dim = -1)
        x = global_add_pool(x, batch)
        return self.mlp(x)


class GPS(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args['hidden_dim'])
        self.pe_input_dim = int(args['pe_dim_input'])
        self.pe_output_dim = int(args['pe_dim_output'])
        self.node_emb = NodeEncoder((self.hidden_dim)//7)
        self.edge_emb = EdgeEncoder((self.hidden_dim+self.pe_output_dim)//2)
        self.convs = ModuleList()
        self.pe_lin = Linear(self.pe_input_dim, self.pe_output_dim)
        self.pe_norm = BatchNorm1d(self.pe_input_dim)
        for _ in range(int(args['num_layers'])):
            nn = Sequential(
                Linear(self.hidden_dim+self.pe_output_dim, self.hidden_dim+self.pe_output_dim),
                BatchNorm1d(self.hidden_dim+self.pe_output_dim),
                ReLU(),
                Dropout(args['dropout']),
                Linear(self.hidden_dim+self.pe_output_dim, self.hidden_dim+self.pe_output_dim),
                BatchNorm1d(self.hidden_dim+self.pe_output_dim),
                ReLU(),
                Dropout(args['dropout']),)
            conv = GPSConv(self.hidden_dim+self.pe_output_dim, 
                           GINEConv(nn), heads=4, attn_type=args['attn_type'], attn_kwargs = args['attn_kwargs'])
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if args['attn_type'] == 'performer' else None)

        self.mlp = Sequential(
            Linear(self.hidden_dim+self.pe_output_dim, (self.hidden_dim+self.pe_output_dim) // 2),
            BatchNorm1d((self.hidden_dim+self.pe_output_dim) // 2),
            ReLU(),
            Dropout(args['dropout']),
            Linear((self.hidden_dim+self.pe_output_dim) // 2, (self.hidden_dim+self.pe_output_dim) // 4),
            BatchNorm1d((self.hidden_dim+self.pe_output_dim) // 4),
            ReLU(),
            Dropout(args['dropout']),
            Linear((self.hidden_dim+self.pe_output_dim) // 4, 1),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x_pe = self.pe_lin(x_pe)
        x = self.node_emb(x)
        x = torch.cat((x, x_pe), -1)
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr = edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)
    

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



class BIGCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args['hidden_dim'])
        self.node_emb = NodeEncoder(self.hidden_dim//7)
        self.edge_emb = EdgeEncoder(self.hidden_dim//2)
        self.convs_forward = ModuleList()
        self.convs_backward = ModuleList()
        for _ in range(int(args['num_layers'])):
            conv_forward = GCNConv(self.hidden_dim, self.hidden_dim)
            conv_backward = GCNConv(self.hidden_dim, self.hidden_dim)
            self.convs_forward.append(conv_forward)
            self.convs_backward.append(conv_backward)

        self.mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            BatchNorm1d(self.hidden_dim // 2),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            BatchNorm1d(self.hidden_dim // 4),
            ReLU(),
            Dropout(args['dropout']),
            Linear(self.hidden_dim // 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        reverse_edge_index = edge_index[[1, 0], :]
        for conv_forward, conv_backward in zip(self.convs_forward, self.convs_backward):
            x_forward = conv_forward(x, edge_index)
            x_backward = conv_backward(x, reverse_edge_index)
            x = (x_forward + x_backward) / 2
        x = global_add_pool(x, batch)
        return self.mlp(x)
    
class GPSSE(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args['hidden_dim'])
        self.pe_input_dim = int(args['pe_dim_input'])
        self.pe_output_dim = int(args['pe_dim_output'])
        self.node_emb = NodeEncoder((self.hidden_dim)//7)
        self.edge_emb = EdgeEncoder((self.hidden_dim+self.pe_output_dim*2)//2)
        self.convs = ModuleList()
        self.pe_lin = Linear(self.pe_input_dim, self.pe_output_dim)
        self.pe_norm = BatchNorm1d(self.pe_input_dim)
        self.se_lin = Linear(self.pe_input_dim, self.pe_output_dim)
        self.se_norm = BatchNorm1d(self.pe_input_dim)
        for _ in range(int(args['num_layers'])):
            nn = Sequential(
                Linear(self.hidden_dim+self.pe_output_dim*2, self.hidden_dim+self.pe_output_dim*2),
                BatchNorm1d(self.hidden_dim+self.pe_output_dim*2),
                ReLU(),
                Dropout(args['dropout']),
                Linear(self.hidden_dim+self.pe_output_dim*2, self.hidden_dim+self.pe_output_dim*2),
                BatchNorm1d(self.hidden_dim+self.pe_output_dim*2),
                ReLU(),
                Dropout(args['dropout']),)
            conv = GPSConv(self.hidden_dim+self.pe_output_dim*2, 
                           GINEConv(nn), heads=4, attn_type=args['attn_type'], attn_kwargs = args['attn_kwargs'])
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if args['attn_type'] == 'performer' else None)

        self.mlp = Sequential(
            Linear(self.hidden_dim+self.pe_output_dim*2, (self.hidden_dim+self.pe_output_dim*2) // 2),
            BatchNorm1d((self.hidden_dim+self.pe_output_dim*2) // 2),
            ReLU(),
            Dropout(args['dropout']),
            Linear((self.hidden_dim+self.pe_output_dim*2) // 2, (self.hidden_dim+self.pe_output_dim*2) // 4),
            BatchNorm1d((self.hidden_dim+self.pe_output_dim*2) // 4),
            ReLU(),
            Dropout(args['dropout']),
            Linear((self.hidden_dim+self.pe_output_dim*2) // 4, 1),
        )

    def forward(self, x, pe, se, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x_pe = self.pe_lin(x_pe)
        x_se = self.se_norm(se)
        x_se = self.se_lin(x_se)
        x = self.node_emb(x)
        x = torch.cat((x, x_pe), -1)
        x = torch.cat((x, x_se), -1)
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr = edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)