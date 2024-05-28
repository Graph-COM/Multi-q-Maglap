import torch
from torch import nn
from model.nns import MLPs
from model.gnns import GIN
from utils.activation import MultiheadAttention
from torch.nn import LayerNorm
from torch_geometric.utils import to_dense_batch
from utils.handle_complex import DenseComplexNetwork

class PEAttention(nn.Module):
    # self-attention for PE, i.e., complex array with shape [N, Q, d]
    def __init__(self, pe_config, in_dim, num_heads=4):
        super(PEAttention, self).__init__()
        assert in_dim % num_heads == 0
        self.q_dim = pe_config['q_dim']
        self.projection_q = nn.Linear(in_dim, in_dim, bias=False)
        self.projection_k = nn.Linear(in_dim, in_dim, bias=False)
        self.projection_v = nn.Linear(in_dim, in_dim, bias=False)
        self.num_heads = num_heads


    def forward(self, x, mask):
        # x: [B (batch_size), N (max_num_nodes), Q (num_qs), d (pe_dim), M (hidden_dim)],
        # mask: [B, N]
        x = x.flatten(2, 3) # [B, N, Q*d, M]
        q, k, v = self.projection_q(x), self.projection_k(x), self.projection_v(x) # [B, N, Q*d, M]
        q, k, v = q.unflatten(-1, [self.num_heads, -1]), k.unflatten(-1, [self.num_heads, -1]), \
                  v.unflatten(-1, [self.num_heads, -1]) # [B, N, Q*d, #heads, M / #heads]
        attn_weight = torch.einsum('bndhk, bmdhk->bnmdk', q, k) # [B, N, N, Q*d, #heads]
        attn_mask = torch.zeros_like(mask).float()
        attn_mask[mask] = -torch.inf
        attn_weight = attn_weight + attn_mask.unsqueeze(1).tile([1, attn_weight.size(1), 1])[..., None, None]
        attn_weight = torch.nn.Softmax(dim=2)(attn_weight)
        x = torch.einsum('bnmdh, bndhk->bndhk', attn_weight, v)
        x = x.flatten(-2) # [B, N, Q, d, M]
        x = x.unflatten(2, [self.q_dim, -1])
        return x


class NaivePEEncoder(nn.Module):
    def __init__(self, pe_config, out_dim):
        super(NaivePEEncoder, self).__init__()
        self.pe_dim = pe_config['pe_dim']
        self.q_dim = pe_config['q_dim']
        self.pe_type = pe_config['pe_type']
        pe_in_dim = self.pe_dim * self.q_dim
        if self.pe_type == 'maglap':
            pe_in_dim *= 2
        self.linear = nn.Linear(pe_in_dim, out_dim)
    def forward(self, x, Lambda, edge_index, batch):
        x = x.flatten(1)
        if self.pe_type == 'maglap':
            x = torch.cat([x.real, x.imag], dim=-1)
        return self.linear(x)

class PEEncoder(nn.Module):
    def __init__(self, pe_config, out_dim, encoder='mlp', sign_inv=False, attn=False, dropout=0.1):
        super(PEEncoder, self).__init__()
        self.pe_type = pe_config['pe_type']
        self.q_dim = pe_config['q_dim'] if self.pe_type != 'svd' else 2
        self.pe_dim = pe_config['pe_dim']
        self.pe_norm = pe_config['pe_norm']
        self.sign_inv = sign_inv
        self.encoder_name = encoder
        in_dim = 2 if self.pe_type == 'maglap' else 1
        pe_hidden_dim = 16 # TO DO: make this adjustable?
        if self.encoder_name == 'mlp':
            self.pe_encoder = MLPs(in_dim, pe_hidden_dim, pe_hidden_dim, 3)
            self.pe_projection = nn.Linear(in_dim + pe_hidden_dim, pe_hidden_dim - 1)
        elif self.encoder_name == 'gnn':
            self.pe_encoder = nn.ModuleList([nn.Linear(in_dim, pe_hidden_dim), GIN(pe_hidden_dim, 1)])
            self.pe_projection = nn.Linear(in_dim + pe_hidden_dim, pe_hidden_dim - 1)
        elif self.encoder_name == 'spe':
            self.pe_encoder = nn.ModuleList([DenseComplexNetwork(self.pe_dim, self.q_dim, self.pe_type,
                                                                 pe_hidden_dim, norm=self.pe_norm),
                                             GIN(pe_hidden_dim, 2)]) # TO DO: make this tunable
            self.pe_projection = nn.Linear(pe_hidden_dim, out_dim)
        if attn:
            self.attn = PEAttention(pe_config, pe_hidden_dim)
        else:
            self.attn = None
        self.norm = LayerNorm(pe_hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        # self.readout_mlp = MLPs(self.q_dim * self.pe_dim * pe_hidden_dim, 256, out_dim, 3)
        self.readout_mlp = nn.Linear(self.q_dim * self.pe_dim * pe_hidden_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x, Lambda, edge_index, batch):
        if self.pe_type == 'maglap' and self.encoder_name != 'spe': # for stable pe, do nothing
            # x = x.unflatten(-1, (self.q_dim, self.pe_dim, 2)) # [N, Q, pe_dim, 2]
            x = torch.cat([x.real.unsqueeze(-1), x.imag.unsqueeze(-1)], dim=-1)
        elif self.pe_type == 'lap' and self.encoder_name != 'spe':  # laplacian pe
            # x = x.unflatten(-1, (1, self.pe_dim, 1))
            x = x.unsqueeze(1).unsqueeze(-1)
            Lambda = Lambda.unsqueeze(1)
        elif self.pe_type == 'svd':
            x = x.unsqueeze(-1)
            Lambda = Lambda.unsqueeze(1).tile([1, 2, 1])

        # pe encoder
        if self.encoder_name == 'mlp':
            x_emb = self.pe_encoder(x) # [N, Q, pe_dim, pe_hidden_dim]
            if self.sign_inv:
                x_emb += self.pe_encoder(-x)
        elif self.encoder_name == 'gnn':
            x_emb = self.pe_encoder[0](x)
            x_emb = self.pe_encoder[1](x_emb, edge_index, None)
            if self.sign_inv:
                x_neg = self.pe_encoder[0](-x)
                x_neg = self.pe_encoder[1](x_neg, edge_index, None)
                x_emb += x_neg
        elif self.encoder_name == 'spe':
            x_emb = self.pe_encoder[0](x, Lambda, batch) # [B, N_max, N_max, pe_hidden_dim]
            _, mask = to_dense_batch(x, batch)
            x_emb = x_emb[mask]  # [N_sum, N_max, pe_hidden_dim]
            x_emb = self.pe_encoder[1](x_emb, edge_index, None)
            x_emb = (x_emb * mask[batch].unsqueeze(-1)).sum(1) # [N_sum, pe_hidden_dim]
            x_emb = self.norm(x_emb)
            x_emb = self.dropout(x_emb)
            return self.act(self.pe_projection(x_emb)) # directly return

        x = torch.cat([x, x_emb], dim=-1) # [N, Q, pe_dim, pe_hidden_dim + 2]
        x = self.pe_projection(x) # [N, Q, pe_dim, pe_hidden_dim - 1]

        # concat eigenvalues
        # Lambda = Lambda.unflatten(-1, (self.q_dim, self.pe_dim)) # [B, Q, pe_dim]
        x = torch.cat([x, Lambda[batch].unsqueeze(-1)], dim=-1) # [N, Q, pe_dim, pe_hidden_dim]

        # a global self-attention layer
        x = self.norm(x)
        if self.attn is not None:
            x_b, mask = to_dense_batch(x, batch)
            x_b = self.attn(x_b, ~mask)
            x = x + x_b[mask]

        # readout
        x = x.flatten(1) # [N, Q * pe_dim * pe_hidden_dim]
        x = self.dropout(x)
        x = self.readout_mlp(x)
        x = self.act(x)

        return x


