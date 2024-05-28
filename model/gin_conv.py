import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from torch.nn import Linear

class GINConvEquivariant(MessagePassing):
    def __init__(self, mlp, edge_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConvEquivariant, self).__init__(aggr = "add")

        # self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.mlp = mlp # must be a vector mlp
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        # edge_attr is two dimensional after augment_edge transformation
        if edge_dim is not None:
            if isinstance(self.mlp, torch.nn.Sequential):
                mlp = self.mlp[0]
            if hasattr(mlp, 'in_features'):
                in_channels = mlp.in_features
            elif hasattr(mlp, 'in_channels'):
                in_channels = mlp.in_channels
            else:
                raise ValueError("Could not infer input channels from `mlp`.")
            self.pe_hidden_dim = in_channels
            self.lin = Linear(edge_dim, in_channels ** 2)
        else:
            self.lin = None
    def forward(self, x, edge_index, edge_attr):
        # x is equivariant features, and edge_atrr is invariant features
        edge_embedding = self.lin(edge_attr)
        message = self.propagate(edge_index, x=x.flatten(1), edge_attr=edge_embedding)
        out = (1 + self.eps) *x + message.unflatten(-1, [self.pe_hidden_dim, -1])
        out = self.mlp(out.transpose(1, 2)).transpose(1, 2)

        return out

    def message(self, x_j, edge_attr):
        # x_j: [E, pe_dim, hidden_dim], edge_attr: [E, ]
        # return F.relu(x_j + edge_attr)
        x_j = x_j.unflatten(1, [self.pe_hidden_dim, -1])
        edge_attr = edge_attr.unflatten(1, [self.pe_hidden_dim, -1])
        # return x_j @ edge_attr
        return torch.einsum('ehd, ejh->ejd', x_j, edge_attr).flatten(1)

    def update(self, aggr_out):
        return aggr_out