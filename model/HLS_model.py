import torch
from torch.nn import BatchNorm1d, Linear, ModuleList, ReLU, Sequential

from torch_geometric.nn import GINEConv, GCNConv
from model.gps_conv import GPSConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.attention import PerformerAttention
from utils.handle_complex import ComplexHandler
from model.nns import MLPs
from torch_geometric.utils import to_dense_batch
#from torch_geometric_signed_directed.nn import MSConv
#from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer


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
        #x_embedding = 0
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
        '''edge_embedding = 0
        for i in range(edge_attr.shape[1]):
            edge_embedding += self.edge_embedding_list[i](edge_attr[:,i])'''
        edge_embedding_list = []
        for i in range(edge_attr.shape[1]):
            edge_embedding_list.append(self.edge_embedding_list[i](edge_attr[:,i]))
        edge_embedding = torch.cat(edge_embedding_list, dim = 1)
        return edge_embedding   

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
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),)
            conv = GINEConv(nn)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            BatchNorm1d(self.hidden_dim // 2),
            ReLU(),
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            BatchNorm1d(self.hidden_dim // 4),
            ReLU(),
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
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            BatchNorm1d(self.hidden_dim // 4),
            ReLU(),
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
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),)
            nn_backward = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),)
            conv_forward = GINEConv(nn_forward)
            conv_backward = GINEConv(nn_backward)
            self.convs_forward.append(conv_forward)
            self.convs_backward.append(conv_backward)

        self.mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            BatchNorm1d(self.hidden_dim // 2),
            ReLU(),
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            BatchNorm1d(self.hidden_dim // 4),
            ReLU(),
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
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            BatchNorm1d(self.hidden_dim // 4),
            ReLU(),
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
        self.pe_input_dim = 2 * args['q_dim'] * int(args['pe_dim_input']) if args['pe_type'] == 'maglap' else int(args['pe_dim_input'])
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
                Linear(self.hidden_dim+self.pe_output_dim, self.hidden_dim+self.pe_output_dim),
                BatchNorm1d(self.hidden_dim+self.pe_output_dim),
                ReLU(),)
            conv = GPSConv(self.hidden_dim+self.pe_output_dim,
                           GINEConv(nn), heads=4, dropout=args['dropout'], attn_type=args['attn_type'])
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if args['attn_type'] == 'performer' else None)

        #self.mlp = Sequential(
        #    Linear(self.hidden_dim+self.pe_output_dim, (self.hidden_dim+self.pe_output_dim) // 2),
        #    BatchNorm1d((self.hidden_dim+self.pe_output_dim) // 2),
        #    ReLU(),
        #    Linear((self.hidden_dim+self.pe_output_dim) // 2, (self.hidden_dim+self.pe_output_dim) // 4),
        #    BatchNorm1d((self.hidden_dim+self.pe_output_dim) // 4),
        #    ReLU(),
        #    Linear((self.hidden_dim+self.pe_output_dim) // 4, 1),
        #)
        self.out_dim = self.hidden_dim + self.pe_output_dim

    def forward(self, x, edge_index, edge_attr, batch):
        #x_pe = self.pe_norm(pe)
        #x_pe = self.pe_lin(x_pe)
        #x = self.node_emb(x)
        #x = torch.cat((x, pe), -1)
        #edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr = edge_attr)
        return x
        #x = global_add_pool(x, batch)
        #return self.mlp(x)

    @property
    def out_dims(self) -> int:
        return self.out_dim
    

class GPSInvariant(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args['hidden_dim'])
        self.pe_input_dim = 2 * args['q_dim'] * int(args['pe_dim_input']) if args['pe_type'] == 'maglap' else int(
            args['pe_dim_input'])
        self.pe_output_dim = int(args['pe_dim_output'])
        self.node_emb = NodeEncoder((self.hidden_dim) // 7)
        self.edge_emb = EdgeEncoder((self.hidden_dim) // 2)
        self.pe_edge_emb = torch.nn.Linear(2 * args['q_dim'], self.hidden_dim)
        self.convs = ModuleList()
        self.pe_lin = Linear(self.pe_input_dim, self.pe_output_dim)
        self.pe_norm = BatchNorm1d(self.pe_input_dim)
        self.complex_handler = ComplexHandler(pe_dim=self.pe_input_dim, q_dim=args['q_dim'])
        for _ in range(int(args['num_layers'])):
            nn = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(),
                Linear(self.hidden_dim, self.hidden_dim),
                BatchNorm1d(self.hidden_dim),
                ReLU(), )
            conv = GPSConv(self.hidden_dim,
                           GINEConv(nn), heads=4, dropout=args['dropout'], attn_type=args['attn_type'])
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if args['attn_type'] == 'performer' else None)

        # handle pe-based invariant attn
        self.eigval_encoder = MLPs(1, 32, 8, 3)
        self.attn_bias_projection = Linear(2 * args['q_dim'] * 8, 4)
        self.complex_handler = ComplexHandler(pe_dim=self.pe_input_dim, q_dim=args['q_dim'])

        self.out_dim = self.hidden_dim

    def forward(self, x, pe, Lambda, edge_index, edge_attr, batch):
        # x_pe = self.pe_norm(pe)
        # x_pe = self.pe_lin(x_pe)
        # encode pe into attn bias
        pe, _ = to_dense_batch(pe, batch)
        Lambda = Lambda.unflatten(-1, (self.complex_handler.q_dim, -1)) # [B, Q, pe_dim]
        Lambda = self.eigval_encoder(Lambda.unsqueeze(-1))  # [B, Q, pe_dim, 8]
        attn_bias = self.complex_handler.weighted_gram_matrix_batched(pe, Lambda) # [B, N, N, q_dim * 8]
        attn_bias = self.attn_bias_projection(attn_bias) # [B, N, N, #heads]
        attn_bias = torch.transpose(attn_bias, 1, -1).flatten(0, 1) # [B*#heads, N, N]
        for conv in self.convs:
            x = conv(x, edge_index, batch, attn_bias=attn_bias, edge_attr = edge_attr)
        return x

    @property
    def out_dims(self) -> int:
        return self.out_dim



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
