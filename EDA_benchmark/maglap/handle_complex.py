import torch
from torch import nn
from models.base_model import MLPs
from torch_geometric.utils import to_dense_batch
# in all the following, complex vector x are represented by real and imag parts, which is
# x = [RE_1, IM_1, RE_2, IM_2, ..., RE_d, IM_d]

class ComplexHandler:
    def __init__(self, pe_dim, q_dim, pe_type):
        self.pe_dim = pe_dim
        self.q_dim = q_dim
        #self.pe_actual_dim = int(pe_dim / q_dim / 2)
        self.pe_type = pe_type

    def merge_real_imag(self, x):
        # input x: [***, pe_dim]
        # output x: [***, q_dim, pe_actual_dim]
        if self.pe_type == 'maglap':
            x = x.unflatten(-1, (self.q_dim, self.pe_dim, 2))
            x = x[..., 0] + 1j * x[..., 1]
        else:# laplacian pe
            x = x.unflatten(-1, (1, self.pe_dim))
            x = x + 1j * 0. # treated as complex number
        return x

    def get_magnitude(self, x):
        # input x: [***, pe_dim]
        # output x: [***, pe_dim / 2] for maglap or [***, pe_dim] for lap
        # if self.pe_type == 'maglap':
        #   x = x.unflatten(-1, (self.q_dim, self.pe_actual_dim, 2))
        #    x = torch.sqrt(torch.sum(x ** 2, -1))
        #else:
        #    x = x.unflatten(-1, (1, self.pe_dim))
        #    x = torch.abs(x)
        return torch.abs(x)

    def gram_matrix_batched(self, x):
        # input x: [B, N, pe_dim], where pe_dim = q_dim * pe_actual_dim * 2
        # output x: [B, N, N, q_dim], doing inner product along pe_actual_dim axis
        # x = self.merge_real_imag(x) # [B, N, q_dim, pe_actual_dim]
        gram = torch.einsum('bnqd, bmqd->bnmq', torch.conj(x), x)
        return torch.cat([torch.real(gram), torch.imag(gram)], dim=-1)

    def pointwise_gram_matrix_batched(self, x):
        # input x: [B, N, pe_dim], where pe_dim = q_dim * pe_actual_dim * 2
        # output x: [B, N, N, q_dim * pe_actual_dim * 3]
        N = x.size(1)
        # x = self.merge_real_imag(x) # [B, N, q_dim, pe_actual_dim]
        gram = torch.einsum('bnqd, bmqd->bnmqd', torch.conj(x), x).flatten(-2) # [B, N, N, q_dim * pe_actual_dim]
        x_mag = (torch.conj(x) * x).flatten(-2) # [B, N, q_dim * pe_actual_dim]
        gram = torch.cat([gram , x_mag.unsqueeze(1).tile(1, N, 1, 1), x_mag.unsqueeze(2).tile(1, 1, N, 1)],
                         dim=-1) # [B, N, N, q_dim * pe_actual_dim * 3]
        return torch.cat([torch.real(gram), torch.imag(gram)], dim=-1)


    def weighted_gram_matrix_batched(self, x, weight):
        # input x: [B, N, pe_dim], weight: [B, q_dim, pe_actual_dim, channels], where pe_dim = q_dim * pe_actual_dim * 2
        # output x: [B, N, N, 2*q_dim * channels], doing inner product along pe_actual_dim axis
        # x = self.merge_real_imag(x) # [B, N, q_dim, pe_actual_dim]
        gram = torch.einsum('bnqd, bmqd, bqdc->bnmqc', x, torch.conj(x), weight.type(torch.complex64))
        gram = gram.flatten(3)
        return torch.cat([torch.real(gram), torch.imag(gram)], dim=-1)

    def weighted_gram_matrix_sparse(self, x, edge_index, weight=None):
        # input x: [BN, pe_dim], edge_index: [2, BE], weight: [BE, q_dim, pe_actual_dim, channels]
        # output x: [BE, q_dim, channels]
        # x = self.merge_real_imag(x)  # [BN, Q, pe_dim]
        if x.ndim == 3:
            if weight is not None:
                x = weight * x[edge_index[0]].unsqueeze(-1) * torch.conj(x[edge_index[1]]).unsqueeze(
                    -1)  # [BE, Q, pe_actual_dim, 8]
            else:
                x = x[edge_index[0]].unsqueeze(-1) * torch.conj(x[edge_index[1]]).unsqueeze(
                    -1)  # [BE, Q, pe_actual_dim, 8]
            x = x.sum(2).flatten(1)
        elif x.ndim == 4: # [BN, pe_hidden_dim, Q, pe_actual_dim]
            if weight is not None:
                weight = weight.transpose(1, 2).transpose(1, -1) # [BE, pe_hiddem_dim, Q, pe_actual_dim]
                x = weight * x[edge_index[0]] * torch.conj(x[edge_index[1]])  # [BE, pe_hidden_dim, Q, pe_actual_dim]
            else:
                x = x[edge_index[0]] * torch.conj(x[edge_index[1]])   # [BE, pe_hiddem_dim, Q, pe_actual_dim]
            x = x.sum(-1).flatten(1) # [BE, pe_hidden_dim * Q]
        return torch.cat([x.real, x.imag], dim=-1)

    def pointwise_correlation_matrix_batched(self, x, weight=None):
        # input x: [B, N, pe_dim], weight: [B, q_dim, pe_actual_dim], where pe_dim = q_dim * pe_actual_dim * 2
        # output x: [B, N, N, 4*q_dim*pe_actual_dim]
        N = x.size(1)
        x_norm = self.get_magnitude(x) # [B, N, q_dim, pe_actual_dim]
        x_norm = torch.cat([x_norm.unsqueeze(1).tile([1, N, 1, 1, 1]), x_norm.unsqueeze(2).tile([1, 1, N, 1, 1])], dim=-1)
        # x = self.merge_real_imag(x)  # [B, N, q_dim, pe_actual_dim]
        x = torch.einsum('bnqd, bmqd->bnmqd', x, torch.conj(x)) # [BE, q_dim, pe_actual_dim]
        if weight is not None:
            weight = weight.unsqueeze(1).unsqueeze(1)
            weight = torch.tile(weight, [1, N, N, 1, 1])
            x = torch.cat([x, x_norm, weight], dim=-1)  # [B, N, N, q_dim, 4 * pe_actual_dim]
        else:
            x = torch.cat([x, x_norm], dim=-1)  # [B, N, N, q_dim, 3 * pe_actual_dim]
        x = x.flatten(3)
        return torch.cat([x.real, x.imag], dim=-1)

    def pointwise_correlation_matrix_sparse(self, x, edge_index, weight=None):
        # input x: [BN, pe_dim], edge_index: [2, BE], weight: [BE, q_dim, pe_actual_dim]
        # output x: [BE, q_dim, pe_actual_dim, 3 or 4]
        x_norm = self.get_magnitude(x)  # [BN, q_dim, pe_actual_dim]
        x_norm = torch.cat([x_norm[edge_index[0]], x_norm[edge_index[1]]], dim=-1)  # [BE, q_dim, 2 * pe_actual_dim]
        # x = self.merge_real_imag(x)  # [BN, q_dim, pe_actual_dim]
        x = x[edge_index[0]] * torch.conj(x[edge_index[1]])  # [BE, q_dim, pe_actual_dim]
        if weight is not None:
            x = torch.cat([x, x_norm, weight], dim=-1)  # [BE, q_dim, 4 * pe_actual_dim]
        else:
            x = torch.cat([x, x_norm], dim=-1)  # [BE, q_dim, 3 * pe_actual_dim]
        x = x.flatten(1)
        return torch.cat([x.real, x.imag], dim=-1)

    def simple_pointwise_correlation_matrix_sparse(self, x, edge_index, weight=None):
        # input x: [BN, pe_dim], edge_index: [2, BE], weight: [BE, q_dim, pe_actual_dim]
        # output x: [BE, q_dim, pe_actual_dim, 3 or 4]
        x_norm = self.get_magnitude(x).unsqueeze(-1)  # [BN, q_dim, pe_actual_dim, 1]
        x_norm = torch.cat([x_norm[edge_index[0]], x_norm[edge_index[1]]], dim=-1)  # [BE, q_dim, pe_actual_dim, 2]
        # x = self.merge_real_imag(x)  # [BN, q_dim, pe_actual_dim]
        x = x[edge_index[0]] * torch.conj(x[edge_index[1]])  # [BE, q_dim, pe_actual_dim]
        x = x.unsqueeze(-1) # [BE, q_dim, pe_actual_dim, 1]
        x = torch.cat([x.real, x.imag], dim=-1) # [BE, q_dim, pe_actual_dim, 2]
        if weight is not None:
            x = torch.cat([x, x_norm, weight.unsqueeze(-1)], dim=-1)  # [BE, q_dim, pe_actual_dim, 5]
        else:
            x = torch.cat([x, x_norm], dim=-1)  # [BE, q_dim, pe_actual_dim, 4]
        return x


    def simple_pointwise_correlation_matrix_batched(self, x, weight=None):
        #N = x.size(1)
        #x_norm = self.get_magnitude(x).unsqueeze(-1)  # [B, N, q_dim, pe_actual_dim, 1]
        #x_norm = torch.cat([x_norm.unsqueeze(1).tile([1, N, 1, 1, 1, 1]), x_norm.unsqueeze(2).tile([1, 1, N, 1, 1, 1])],
        #                   dim=-1) # [B, N, N, q_dim, pe_actual_dim, 2]
        #x = self.merge_real_imag(x)  # [B, N, q_dim, pe_actual_dim]
        #x = torch.einsum('bnqd, bmqd->bnmqd', x, torch.conj(x))  # [B, N, N, q_dim, pe_actual_dim]
        #x = x.unsqueeze(-1) # [B, N, N, q_dim, pe_actual_dim, 1]
        #x = torch.cat([x.real, x.imag], dim=-1) # [B, N, N, q_dim, pe_actual_dim, 2]
        #if weight is not None:
        #    weight = weight.unsqueeze(1).unsqueeze(1)
        #    weight = torch.tile(weight, [1, N, N, 1, 1])
        #    x = torch.cat([x, x_norm, weight.unsqueeze(-1)], dim=-1)  # [B, N, N, q_dim, pe_actual_dim, 5]
        #else:
        #    x = torch.cat([x, x_norm], dim=-1)  # [B, N, N, q_dim, pe_actual_dim, 4]
        # input x: [B, N, pe_dim], weight: [B, q_dim, pe_actual_dim], where pe_dim = q_dim * pe_actual_dim * 2
        # output x: [B, N, N, 4*q_dim*pe_actual_dim]
        N = x.size(1)
        x_norm = self.get_magnitude(x)  # [B, N, q_dim, pe_actual_dim]
        x_norm = torch.cat([x_norm.unsqueeze(1).tile([1, N, 1, 1, 1]), x_norm.unsqueeze(2).tile([1, 1, N, 1, 1])],
                           dim=-1)
        # x = self.merge_real_imag(x)  # [B, N, q_dim, pe_actual_dim]
        x = torch.einsum('bnqd, bmqd->bnmqd', x, torch.conj(x))  # [B, N, N q_dim, pe_actual_dim]
        x = torch.cat([x.real, x.imag], dim=-1) # [B, N, N, q_dim, 2 * pe_actual_dim]
        if weight is not None:
            weight = weight.unsqueeze(1).unsqueeze(1)
            weight = torch.tile(weight, [1, N, N, 1, 1])
            x = torch.cat([x, x_norm, weight], dim=-1)  # [B, N, N, q_dim, 5 * pe_actual_dim]
        else:
            x = torch.cat([x, x_norm], dim=-1)  # [B, N, N, q_dim, 4 * pe_actual_dim]
        return x.unflatten(-1, [5, self.pe_actual_dim]).transpose(-1, -2)




class SparseComplexNetwork(torch.nn.Module):
    def __init__(self, pe_dim, q_dim, pe_type, out_dim, network_type='spe', norm=None):
        super(SparseComplexNetwork, self).__init__()
        assert network_type == 'pointwise' or network_type == 'spe' or network_type=='pointwise-simple'
        self.pe_type = pe_type
        self.complex_handler = ComplexHandler(pe_dim, q_dim, pe_type=pe_type)
        self.network_type = network_type
        #pe_actual_dim = int(pe_dim / q_dim / 2) if pe_type == 'maglap' else pe_dim
        if network_type == 'spe':
            #eigval_dim = 16
            eigval_dim = 8 # for HLS only
            self.eigval_encoder = MLPs(1, 32, eigval_dim, 3, norm=norm)
            self.readout = nn.Linear(2 * q_dim * eigval_dim, out_dim)
        elif network_type == 'pointwise':
            self.readout = MLPs(8 * pe_dim * q_dim, pe_dim * q_dim, out_dim, 3)
        elif network_type == 'pointwise-simple':
            self.pe_encoder = MLPs(5, 32, 8, 3)
            self.readout = MLPs(8 * q_dim, 4 * q_dim, out_dim, 3)

    def forward(self, x, Lambda, edge_index, batch):
        # Lambda = Lambda.unflatten(-1, (self.complex_handler.q_dim, -1))
        if self.pe_type == 'lap':
            x = x.unsqueeze(1) + 0 * 1j # [BE, 1, pe_dim]
            Lambda = Lambda.unsqueeze(1) # [B, 1, pe_dim]
        if self.network_type == 'spe':
            Lambda = self.eigval_encoder(Lambda.unsqueeze(-1))  # [B, Q, pe_dim, 8]
            Lambda = Lambda[batch[edge_index[0]]]  # [BE, Q, pe_dim, 8]
            x = self.complex_handler.weighted_gram_matrix_sparse(x, edge_index, Lambda)
        elif self.network_type == 'pointwise':
            Lambda = Lambda[batch[edge_index[0]]]  # [BE, Q, pe_dim, 8]
            x = self.complex_handler.pointwise_correlation_matrix_sparse(x, edge_index, Lambda)
        elif self.network_type == 'pointwise-simple':
            Lambda = Lambda[batch[edge_index[0]]]  # [BE, Q, pe_dim, 8]
            x = self.complex_handler.simple_pointwise_correlation_matrix_sparse(x, edge_index, Lambda)
            x = self.pe_encoder(x) # [BE, Q, pe_dim, 8]
            x = x.sum(2).flatten(1) # [BE, Q * 8]
        return self.readout(x)



class DenseComplexNetwork(torch.nn.Module):
    def __init__(self, pe_dim, q_dim, pe_type, out_dim, network_type='spe', norm=None):
        super(DenseComplexNetwork, self).__init__()
        assert network_type == 'pointwise' or network_type == 'spe' or network_type == 'pointwise-simple'
        self.pe_type = pe_type
        self.complex_handler = ComplexHandler(pe_dim, q_dim, pe_type=pe_type)
        self.network_type = network_type
        self.pe_type = pe_type
        #pe_actual_dim = int(pe_dim / q_dim / 2) if pe_type == 'maglap' else pe_dim
        if network_type == 'spe':
            eigval_dim = 16
            self.eigval_encoder = MLPs(1, 32, eigval_dim, 3, norm=norm)
            self.readout = MLPs(2 * q_dim * eigval_dim, q_dim * eigval_dim, out_dim, 2, norm=norm)
        elif network_type == 'pointwise':
            self.readout = MLPs(8 * pe_dim * q_dim, pe_dim * q_dim, out_dim, 3)
        elif network_type == 'pointwise-simple':
            self.pe_encoder = MLPs(5, 4, 4, 2)
            self.readout = MLPs(4 * q_dim, q_dim, out_dim, 3)

    def forward(self, x, Lambda, batch):
        x, _ = to_dense_batch(x, batch) # [B, N, Q, pe_dim]
        if self.pe_type == 'lap':
            x = x.unsqueeze(-2) + 0 * 1j # [B, N, 1, pe_dim]
            Lambda = Lambda.unsqueeze(1)  # [B, 1, pe_dim]
        # Lambda = Lambda.unflatten(-1, (self.complex_handler.q_dim, -1))  # [B, q_dim, pe_actual_dim, 1]
        if self.network_type == 'spe':
            Lambda = self.eigval_encoder(Lambda.unsqueeze(-1))  # [B, q_dim, pe_actual_dim, c]
            x = self.complex_handler.weighted_gram_matrix_batched(x, Lambda)
        elif self.network_type == 'pointwise':
            x = self.complex_handler.pointwise_correlation_matrix_batched(x, Lambda)
        elif self.network_type == 'pointwise-simple':
            x = self.complex_handler.simple_pointwise_correlation_matrix_batched(x, Lambda)
            x = self.pe_encoder(x)  # [B, N, N, Q, pe_dim, 8]
            x = x.sum(-2).flatten(3)  # [B, N, N, Q * 8]

        return self.readout(x)



class SparseComplexNetworkEq(torch.nn.Module):
    def __init__(self, pe_dim, q_dim, pe_hidden_dim, pe_type, out_dim, network_type='spe'):
        super(SparseComplexNetworkEq, self).__init__()
        assert network_type == 'pointwise' or network_type == 'spe' or network_type=='pointwise-simple'
        self.complex_handler = ComplexHandler(pe_dim, q_dim, pe_type=pe_type)
        self.network_type = network_type
        pe_actual_dim = int(pe_dim / q_dim / 2) if pe_type == 'maglap' else pe_dim
        if network_type == 'spe':
            self.eigval_encoder = MLPs(1, 32, pe_hidden_dim, 3)
            self.readout = nn.Linear(2 * q_dim * pe_hidden_dim, out_dim)
        #elif network_type == 'pointwise':
        #    self.readout = MLPs(8 * pe_actual_dim * q_dim, pe_actual_dim * q_dim, out_dim, 3)
        #elif network_type == 'pointwise-simple':
        #    self.pe_encoder = MLPs(5, 32, 8, 3)
        #    self.readout = MLPs(8 * q_dim, 4 * q_dim, out_dim, 3)

    def forward(self, x, Lambda, edge_index, batch):
        Lambda = Lambda.unflatten(-1, (self.complex_handler.q_dim, -1))
        if self.network_type == 'spe':
            Lambda = self.eigval_encoder(Lambda.unsqueeze(-1))  # [B, Q, pe_dim, 8]
            Lambda = Lambda[batch[edge_index[0]]]  # [BE, Q, pe_dim, 8]
            x = self.complex_handler.weighted_gram_matrix_sparse(x, edge_index, Lambda)
        elif self.network_type == 'pointwise':
            Lambda = Lambda[batch[edge_index[0]]]  # [BE, Q, pe_dim, 8]
            x = self.complex_handler.pointwise_correlation_matrix_sparse(x, edge_index, Lambda)
        elif self.network_type == 'pointwise-simple':
            Lambda = Lambda[batch[edge_index[0]]]  # [BE, Q, pe_dim, 8]
            x = self.complex_handler.simple_pointwise_correlation_matrix_sparse(x, edge_index, Lambda)
            x = self.pe_encoder(x) # [BE, Q, pe_dim, 8]
            x = x.sum(2).flatten(1) # [BE, Q * 8]
        return self.readout(x)


def complex_norm(x):
    # x: [***, 2d]
    x = torch.transpose(x, 0, -1)
    d = x.size(0)
    norm = torch.sqrt(x[0:int(d/2)] ** 2 + x[int(d/2):] ** 2)
    return torch.transpose(norm, 0, -1)

def pairwise_angles(x):
    # x: [B, N, d]
    d, N = x.size(-1), x.size(1)
    x1 = x[:, :, 0:int(d/2)]
    x2 = x[:, :, int(d/2):]
    x1 = x1.unsqueeze(2).expand([-1, -1, N, -1]) # real part
    x2 = x2.unsqueeze(2).expand([-1, -1, N, -1]) # img part
    # for node pair (i, j)
    x1x1 = x1 * x1.transpose(1, 2) # real part of i * real part of j
    x2x2 = x2 * x2.transpose(1, 2) # imag part of i * imag part of j
    x1x2 = x1 * x2.transpose(1, 2) # real part of i * imag part of j
    x2x1 = x2 * x1.transpose(1, 2) # imag part of i * real part of j
    return x1x1 + x2x2, x1x2 - x2x1 # real part and img part of a phase invariant complex number determined by i and j
