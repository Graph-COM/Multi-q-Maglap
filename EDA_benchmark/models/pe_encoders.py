import torch
from torch import nn
from models.base_model import MLPs
from torch_geometric.nn import GINConv
from torch.nn import LayerNorm
from maglap.handle_complex import DenseComplexNetwork
from torch_geometric.utils import to_dense_batch

# this GIN is for pe_projection only
class GIN(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                #nn.BatchNorm1d(self.hidden_dim)
            )
            conv = GINConv(mlp, node_dim=0) # noed_dim = 0 to deal with arbitrary shape pe
            self.convs.append(conv)
            #self.norms.append(InstanceNorm(self.hidden_dim))


    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            #x = self.norms[i](x, batch)
        return x



class PEEmbedderWrapper(nn.Module):
    def __init__(self, args):
        super(PEEmbedderWrapper, self).__init__()
        if args['pe_embedder']['name'] == 'naive':
            self.embedder = NaivePEEmbedder(args)
        elif args['pe_embedder']['name'] in ['regular', 'spe']:
            self.embedder = PEEmbedder(args)
        else:
            raise Exception("pe_embedder name unknown!")

    def forward(self, pe, Lambda, edge_index, batch):
        return self.embedder(pe, Lambda, edge_index, batch)

class NaivePEEmbedder(nn.Module):
    def __init__(self, args):
        super(NaivePEEmbedder, self).__init__()
        self.pe_type = args['pe_type']
        self.pe_dim = args[args['pe_type'][:3] + '_pe_dim_input']
        self.q_dim = args.get('q_dim')
        self.q_dim = 1 if self.q_dim is None else self.q_dim
        out_dim = args[args['pe_type'][:3] + '_pe_dim_output']
        pe_in_dim = self.pe_dim * self.q_dim
        if self.pe_type == 'maglap':
            pe_in_dim *= 2
        self.linear = nn.Linear(pe_in_dim, out_dim)
    def forward(self, x, Lambda, edge_index, batch):
        x = x.flatten(1)
        if self.pe_type == 'maglap':
            x = torch.cat([x.real, x.imag], dim=-1)
        return self.linear(x)

class PEEmbedder(nn.Module):
    def __init__(self, args, dropout=0.1):
        super(PEEmbedder, self).__init__()
        self.pe_type = args['pe_type']
        self.pe_dim = args[args['pe_type'][:3] + '_pe_dim_input']
        self.q_dim = args.get('q_dim')
        self.q_dim = 1 if self.q_dim is None else self.q_dim
        out_dim = args[args['pe_type'][:3] + '_pe_dim_output']
        self.sign_inv = args['pe_embedder'].get('sign_inv') == 1
        if args['pe_embedder']['name'] == 'spe':
            self.encoder_name = 'spe'
        else:
            self.encoder_name = args['pe_embedder']['local_model']
        in_dim = 2 if self.pe_type == 'maglap' else 1
        #pe_hidden_dim = 16 # TO DO: make this adjustable?
        pe_hidden_dim = 8 # for HLS only
        if self.encoder_name == 'mlp':
            self.pe_encoder = MLPs(in_dim, pe_hidden_dim, pe_hidden_dim, 3)
            self.pe_projection = nn.Linear(in_dim + pe_hidden_dim, pe_hidden_dim - 1)
        elif self.encoder_name == 'gnn':
            self.pe_encoder = nn.ModuleList([nn.Linear(in_dim, pe_hidden_dim), GIN(pe_hidden_dim, 2)])
            self.pe_projection = nn.Linear(in_dim + pe_hidden_dim, pe_hidden_dim - 1)
        elif self.encoder_name == 'spe':
            self.pe_encoder = nn.ModuleList([DenseComplexNetwork(self.pe_dim, self.q_dim, self.pe_type, pe_hidden_dim,
                                                                 norm=args['pe_embedder']['norm']),
                                             GIN(pe_hidden_dim, args['pe_embedder']['k_hops'])])  # TO DO: make this tunable
            self.pe_projection = nn.Linear(pe_hidden_dim, out_dim)
        self.norm = LayerNorm(pe_hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        # self.readout_mlp = MLPs(self.q_dim * self.pe_dim * pe_hidden_dim, 256, out_dim, 3)
        self.readout_mlp = nn.Linear(self.q_dim * self.pe_dim * pe_hidden_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x, Lambda, edge_index, batch):
        if self.pe_type == 'maglap' and self.encoder_name != 'spe':
            # x = x.unflatten(-1, (self.q_dim, self.pe_dim, 2)) # [N, Q, pe_dim, 2]
            x = torch.cat([x.real.unsqueeze(-1), x.imag.unsqueeze(-1)], dim=-1)
        elif self.pe_type == 'lap' and self.encoder_name != 'spe':  # laplacian pe
            # x = x.unflatten(-1, (1, self.pe_dim, 1))
            x = x.unsqueeze(1).unsqueeze(-1)
            Lambda = Lambda.unsqueeze(1)

        # pe encoder
        if self.encoder_name == 'mlp':
            x_emb = self.pe_encoder(x) # [N, Q, pe_dim, pe_hidden_dim]
            if self.sign_inv:
                x_emb += self.pe_encoder(-x)
        elif self.encoder_name == 'gnn':
            x_emb = self.pe_encoder[0](x)
            x_emb = self.pe_encoder[1](x_emb, edge_index)
            if self.sign_inv:
                x_neg = self.pe_encoder[0](-x)
                x_neg = self.pe_encoder[1](x_neg, edge_index)
                x_emb += x_neg
        elif self.encoder_name == 'spe':
                x_emb = self.pe_encoder[0](x, Lambda, batch)
                _, mask = to_dense_batch(x, batch)
                x_emb = x_emb[mask]
                x_emb = self.pe_encoder[1](x_emb, edge_index)
                x_emb = (x_emb * mask[batch].unsqueeze(-1)).sum(1)
                #x_emb = self.dropout(x_emb)
                x_emb = self.pe_projection(x_emb)
                x_emb = self.act(x_emb)
                return x_emb # directly return here

        x = torch.cat([x, x_emb], dim=-1) # [N, Q, pe_dim, pe_hidden_dim + 2]
        x = self.pe_projection(x) # [N, Q, pe_dim, pe_hidden_dim - 1]

        # concat eigenvalues
        # Lambda = Lambda.unflatten(-1, (self.q_dim, self.pe_dim)) # [B, Q, pe_dim]
        x = torch.cat([x, Lambda[batch].unsqueeze(-1)], dim=-1) # [N, Q, pe_dim, pe_hidden_dim]

        # a global self-attention layer
        # not implemented yet
        x = self.norm(x)

        # readout
        x = x.flatten(1) # [N, Q * pe_dim * pe_hidden_dim]
        x = self.dropout(x)
        x = self.readout_mlp(x)
        x = self.act(x)

        return x


