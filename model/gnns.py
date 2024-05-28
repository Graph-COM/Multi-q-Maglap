import torch
from torch import nn
#from torch.nn import MultiheadAttention
from utils.activation import MultiheadAttention
from torch_geometric.nn.conv import GINConv, GINEConv
from model.gin_conv import GINConvEquivariant
from torch_geometric.utils import to_dense_batch
from utils.handle_complex import DenseComplexNetwork, SparseComplexNetwork, SparseComplexNetworkEq
from model.gps_conv import GPSConv
from torch_geometric.nn.norm import InstanceNorm
import torch.nn.functional as F

class Id(nn.Module):
    def __init__(self, out_dim):
        super(Id, self).__init__()
        self.out_dim = out_dim

    def forward(self, x, *kwargs):
        return x

    @property
    def out_dims(self) -> int:
        return self.out_dim

    @property
    def name(self) -> str:
        return 'none'



class Id_PE(nn.Module):
    def __init__(self, out_dim):
        super(Id_PE, self).__init__()
        self.out_dim = out_dim

    def forward(self, x, z, *kwargs):
        return x, z

    @property
    def out_dims(self) -> int:
        return self.out_dim

    @property
    def name(self) -> str:
        return 'none_e'



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
            conv = GINConv(mlp, node_dim=0)
            self.convs.append(conv)
            #self.norms.append(InstanceNorm(self.hidden_dim))

        # handle pe-based invariant attn
        self.out_dim = self.hidden_dim

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            #x = self.norms[i](x, batch)
        return x

    @property
    def out_dims(self) -> int:
        return self.out_dim


class GINE(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
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
                )
            conv = GINEConv(mlp, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))
            #self.edge_encoders.append(torch.nn.Linear(hidden_dim, hidden_dim))
            #self.norms.append(InstanceNorm(self.hidden_dim))

        # handle pe-based invariant attn
        self.out_dim = self.hidden_dim

    def forward(self, x, edge_index, edge_attr, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr = edge_attr)
            x = self.norms[i](x)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

    @property
    def out_dims(self) -> int:
        return self.out_dim






class GraphSage(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
            self.norms.append(InstanceNorm(self.hidden_dim))

        # handle pe-based invariant attn
        self.out_dim = self.hidden_dim

    def forward(self, x, pe, Lambda, edge_index, edge_attr, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x, batch)
        return x, pe

    @property
    def out_dims(self) -> int:
        return self.out_dim


class GraphTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads=4):
        super(GraphTransformer, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.out_dim = hidden_dim
        for _ in range(num_layers):
            self.convs.append(MultiheadAttention(hidden_dim, num_heads, batch_first=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        h0 = x
        #h0, mask = to_dense_batch(h0, batch)
        for i, conv in enumerate(self.convs):
            h, mask = to_dense_batch(h0, batch)
            h, _ = conv(h, h, h, key_padding_mask=~mask) # self attention
            #h, _ = conv(h0, h0, h0, ~mask) # self attention
            h = h[mask]
            h = h + h0  # Residual connection.
            h = self.norms[i](h)
                #h = self.norms[i](h.transpose(1, 2)).transpose(1,2)
            h0 = h
        return h

    @property
    def out_dims(self) -> int:
        return self.out_dim

    @property
    def name(self) -> str:
        return 'transformer'



class GraphTransformerInvariant(nn.Module):
    def __init__(self, hidden_dim, pe_dim, q_dim, num_layers, num_heads=4,
                 handle_symmetry='spe', pe_type='maglap'):
        super(GraphTransformerInvariant, self).__init__()
        self.hidden_dim = hidden_dim
        self.pe_input_dim = pe_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.complex_net_dense = torch.nn.ModuleList()
        complex_net_type = handle_symmetry
        for _ in range(num_layers):
            self.convs.append(MultiheadAttention(hidden_dim, num_heads, batch_first=True))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))
            self.complex_net_dense.append(DenseComplexNetwork(pe_dim, q_dim, pe_type, num_heads,
                                                              network_type=complex_net_type))

        # handle pe-based invariant attn
        # self.eigval_encoder = MLPs(1, 32, 8, 3)
        # self.attn_bias_projection = nn.Linear(2 * q_dim * 8, num_heads)
        # self.complex_handler = ComplexHandler(pe_dim=pe_dim, q_dim=q_dim, pe_type=pe_type)
        self.out_dim = self.hidden_dim

    def forward(self, x, z, Lambda, edge_index, edge_attr, batch):
        # x: invariant features, z: equivariant features, Lambda: eigenvalues
        h0 = x
        for i, conv in enumerate(self.convs):
            h, mask = to_dense_batch(h0, batch)
            attn_bias = self.complex_net_dense[i](z, Lambda, batch)
            attn_bias = torch.transpose(attn_bias, 1, -1).flatten(0, 1)  # [B*#heads, N, N]
            # update invariant features
            h, _ = conv(h, h, h, attn_bias=attn_bias, key_padding_mask=~mask) # self attention
            # update equivaraint features
            # z = z
            h = h[mask]
            h = h + h0  # Residual connection.
            h = self.norms[i](h)
            h0 = h
        #h = h[mask] # mask + batch
        return h, z



    @property
    def out_dims(self) -> int:
        return self.out_dim

    @property
    def name(self) -> str:
        return 'transformer'


class GPS(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim))
            conv = GPSConv(self.hidden_dim,
                           GINEConv(mlp, edge_dim=self.hidden_dim), heads=num_heads)
            self.convs.append(conv)
            # self.norms.append(nn.BatchNorm1d(self.hidden_dim))
        self.out_dim = self.hidden_dim

    def forward(self, x, edge_index, edge_attr, batch):
        # x_pe = self.pe_norm(pe)
        # x_pe = self.pe_lin(x_pe)
        # encode pe into attn bias
        # TO DO: attn bias each layers
        # pe, _ = to_dense_batch(pe, batch)
        # Lambda = Lambda.unflatten(-1, (self.complex_handler.q_dim, -1)) # [B, Q, pe_dim]
        # Lambda = self.eigval_encoder(Lambda.unsqueeze(-1))  # [B, Q, pe_dim, 8]
        # attn_bias = self.complex_handler.weighted_gram_matrix_batched(pe, Lambda) # [B, N, N, q_dim * 8]
        # attn_bias = self.attn_bias_projection(attn_bias) # [B, N, N, #heads]

        for i, conv in enumerate(self.convs):
            # invariant edge attr from pe
            x = conv(x, edge_index, batch, attn_bias=None, edge_attr = edge_attr)
            # x = self.norms[i](x) # already did this in gps conv
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

    @property
    def out_dims(self) -> int:
        return self.out_dim

class GPSInvariant(torch.nn.Module):
    def __init__(self, hidden_dim, pe_dim, q_dim, num_layers, pe_type='lap', handle_symmetry='spe', num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pe_input_dim = pe_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.complex_net_sparse = torch.nn.ModuleList()
        self.complex_net_dense = torch.nn.ModuleList()
        complex_net_type = handle_symmetry
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim))
            conv = GPSConv(self.hidden_dim,
                           GINEConv(mlp), heads=num_heads)
            self.convs.append(conv)
            # self.norms.append(nn.BatchNorm1d(self.hidden_dim))
            self.complex_net_sparse.append(SparseComplexNetwork(pe_dim, q_dim, pe_type, hidden_dim,
                                                           network_type=complex_net_type))
            self.complex_net_dense.append(DenseComplexNetwork(pe_dim, q_dim, pe_type, num_heads,
                                                         network_type=complex_net_type))


        # handle pe-based invariant attn
        # self.eigval_encoder = MLPs(1, 32, 8, 3)
        # self.attn_bias_projection = nn.Linear(2 * q_dim * 8, num_heads)
        # self.complex_handler = ComplexHandler(pe_dim=pe_dim, q_dim=q_dim, pe_type=pe_type)
        self.out_dim = self.hidden_dim

    def forward(self, x, pe, Lambda, edge_index, edge_attr, batch):
        # x_pe = self.pe_norm(pe)
        # x_pe = self.pe_lin(x_pe)
        # encode pe into attn bias
        # TO DO: attn bias each layers
        # pe, _ = to_dense_batch(pe, batch)
        # Lambda = Lambda.unflatten(-1, (self.complex_handler.q_dim, -1)) # [B, Q, pe_dim]
        # Lambda = self.eigval_encoder(Lambda.unsqueeze(-1))  # [B, Q, pe_dim, 8]
        # attn_bias = self.complex_handler.weighted_gram_matrix_batched(pe, Lambda) # [B, N, N, q_dim * 8]
        # attn_bias = self.attn_bias_projection(attn_bias) # [B, N, N, #heads]

        # invariant edge attr from pe
        #edge_attr = self.complex_net_sparse[0](pe, Lambda, edge_index, batch)
        # invariant attention from pe
        #attn_bias = self.complex_net_dense[0](pe, Lambda, batch)
        #attn_bias = torch.transpose(attn_bias, 1, -1).flatten(0, 1)  # [B*#heads, N, N]

        for i, conv in enumerate(self.convs):
            # invariant edge attr from pe
            edge_attr_pe = self.complex_net_sparse[i](pe, Lambda, edge_index, batch) + edge_attr
            # invariant attention from pe
            attn_bias = self.complex_net_dense[i](pe, Lambda, batch)
            attn_bias = torch.transpose(attn_bias, 1, -1).flatten(0, 1)  # [B*#heads, N, N]
            x = conv(x, edge_index, batch, attn_bias=attn_bias, edge_attr = edge_attr_pe)
            # x = self.norms[i](x)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x, pe

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


class GINEInvariant(torch.nn.Module):
    def __init__(self, hidden_dim, pe_dim, q_dim, num_layers, pe_type='lap', handle_symmetry='spe'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.edge_encoders = torch.nn.ModuleList()
        complex_net_type = handle_symmetry
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            conv = GINEConv(mlp, self.hidden_dim)
            self.convs.append(conv)
            self.edge_encoders.append(SparseComplexNetwork(pe_dim, q_dim, pe_type, hidden_dim,
                                                           network_type=complex_net_type))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))
            #self.norms.append(InstanceNorm(self.hidden_dim))

        # handle pe-based invariant attn
        self.out_dim = self.hidden_dim

    def forward(self, x, pe, Lambda, edge_index, edge_attr, batch):
        # edge_attr = self.edge_encoders[0](pe, Lambda, edge_index, batch) + edge_attr
        for i, conv in enumerate(self.convs):
            edge_attr_pe = self.edge_encoders[i](pe, Lambda, edge_index, batch) + edge_attr
            x = conv(x, edge_index, edge_attr=edge_attr_pe)
            x = self.norms[i](x)
            if i != len(self.convs) - 1:
                x = F.relu(x)
            # x = x + conv(x, edge_index, edge_attr=edge_attr)
        return x, pe

    @property
    def out_dims(self) -> int:
        return self.out_dim


class GINEEquivariant(torch.nn.Module):
    def __init__(self, hidden_dim, hidden_dim_pe, pe_dim, q_dim, num_layers, pe_type='lap', handle_symmetry='spe'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_pe = hidden_dim_pe
        self.convs = torch.nn.ModuleList()
        self.convs_pe = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.edge_encoders = torch.nn.ModuleList()
        complex_net_type = handle_symmetry
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            conv = GINEConv(mlp, self.hidden_dim)
            self.convs.append(conv)
            conv = GINConvEquivariant(nn.Linear(hidden_dim_pe, hidden_dim_pe), hidden_dim)
            self.convs_pe.append(conv)
            self.edge_encoders.append(SparseComplexNetworkEq(pe_dim, q_dim, hidden_dim_pe, pe_type, hidden_dim,
                                                           network_type=complex_net_type))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))
            #self.norms.append(InstanceNorm(self.hidden_dim))

        # handle pe-based invariant attn
        self.out_dim = self.hidden_dim

    def forward(self, x, pe, Lambda, edge_index, edge_attr, batch):
        # edge_attr = self.edge_encoders[0](pe, Lambda, edge_index, batch) + edge_attr
        z = torch.tile(pe.unsqueeze(1), [1, self.hidden_dim_pe, 1])
        for i, conv in enumerate(zip(self.convs, self.convs_pe)):
            edge_attr_pe = self.edge_encoders[i](z, Lambda, edge_index, batch) + edge_attr
            x = conv[0](x, edge_index, edge_attr=edge_attr_pe)
            z = conv[1](z, edge_index, edge_attr=edge_attr_pe)
            x = self.norms[i](x)
            if i != len(self.convs) - 1:
                x = F.relu(x)
            # x = x + conv(x, edge_index, edge_attr=edge_attr)
        return x, z

    @property
    def out_dims(self) -> int:
        return self.out_dim
