import torch
from torch import nn
from torch.nn import MultiheadAttention
from torch_geometric.nn.conv import GINConv
from torch_geometric.utils import to_dense_batch

class GIN(nn.Module):
    def __init__(self, node_emb_dim, hidden_dim, out_dim, num_layers):
        super(GIN, self).__init__()
        self.convs = nn.ModuleList()
        self.out_dim = out_dim
        for _ in range(num_layers-1):
            mlp = torch.nn.Sequential(nn.Linear(node_emb_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            node_emb_dim = hidden_dim
        mlp = torch.nn.Sequential(nn.Linear(node_emb_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        self.convs.append(GINConv(mlp))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
        return x

    @property
    def out_dims(self) -> int:
        return self.out_dim


class GraphTransformer(nn.Module):
    def __init__(self, node_emb_dim, hidden_dim, out_dim, num_layers, num_heads=4):
        super(GraphTransformer, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.out_dim = out_dim
        self.linear_in = nn.Linear(node_emb_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        for _ in range(num_layers):
            self.convs.append(MultiheadAttention(hidden_dim, num_heads, batch_first=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        h0 = self.linear_in(x)
        h0, mask = to_dense_batch(h0, batch)
        for conv, norm in zip(self.convs, self.norms):
            h, _ = conv(h0, h0, h0, ~mask) # self attention
            h = h + h0  # Residual connection.
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            h0 = h
        h = h[mask] # mask + batch
        h = self.linear_out(h)
        return h

    @property
    def out_dims(self) -> int:
        return self.out_dim
