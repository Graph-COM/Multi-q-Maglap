from torch import nn
import torch
from model.nns import MLPs
from utils.handle_complex import ComplexHandler, SparseComplexNetwork
from torch_geometric.utils import degree
from torch_geometric.nn import global_add_pool, global_mean_pool
from model.pe_encoders import PEEncoder


def sinusoid_position_encoding(
        pos_seq: torch.Tensor,
        hidden_size: int,
        max_timescale: float = 1e4,
        min_timescale: float = 2.,
) -> torch.Tensor:
    """Creates sinusoidal encodings.

    Args:
      pos_seq: Tensor with positional ids.
      hidden_size: `int` dimension of the positional encoding vectors, D
      max_timescale: `int` maximum timescale for the frequency
      min_timescale: `int` minimum timescale for the frequency

    Returns:
      An array of shape [L, D]
    """
    freqs = torch.arange(0, hidden_size, min_timescale).to(pos_seq.device)
    inv_freq = max_timescale**(-freqs / hidden_size)
    sinusoid_inp = torch.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = torch.cat(
        [torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    return pos_emb

class NodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()

        self.emb_dim = emb_dim
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x1 = sinusoid_position_encoding(x[..., 1], self.emb_dim // 2, max_timescale=26) # encoding for i
        x2 = sinusoid_position_encoding(x[..., 2], self.emb_dim // 2, max_timescale=26) # encoding for j
        x_swap = torch.cat([x1, x2], dim=-1)
        x_swap = self.proj(x_swap)
        x_id = sinusoid_position_encoding(x[..., 0], self.emb_dim)
        #return x_swap + x_id # no id embedding
        return x_swap



class GraphClassifier(nn.Module):
    def __init__(self, hidden_dim, gnn_model, pe_model=None):
        super(GraphClassifier, self).__init__()
        #num_node_types = args['node_input_dim']
        #num_edge_types = args['edge_input_dim']
        node_emb_dim = hidden_dim
        # self.node_encoder = NodeEncoder(node_emb_dim)
        self.node_encoder = NodeEncoder(node_emb_dim)
        self.edge_encoder = nn.Embedding(2, node_emb_dim) # for bidirectional edges
        # self.depth_encoder = nn.Embedding(36, node_emb_dim)
        # self.edge_encoder = nn.Embedding(2, node_emb_dim)
        # self.pe_encoder = nn.Linear(pe_dim, node_emb_dim)
        self.pe_encoder = pe_model
        self.gnn_model = gnn_model
        self.output_mlps = MLPs(hidden_dim, hidden_dim, 1, 3)
        # special invariant treatment of maglap pe

    def forward(self, batch):
        # x = self.node_encoder(batch.x) + self.depth_encoder(batch.node_depth[:, 0])
        x = self.node_encoder(batch.x)
        if 'pe' in batch and self.pe_encoder is not None:
            # x = x + self.pe_encoder(batch.pe)
            x = x + self.pe_encoder(batch.pe, batch.Lambda, batch.edge_index, batch.batch)

        # run gnn model
        if 'edge_attr' in batch:
            edge_attr = self.edge_encoder(batch.edge_attr)
        else:
            edge_attr = torch.zeros([batch.edge_index.size(-1), x.size(-1)]).to(x.device)

        if 'dag_rr_edge_index' in batch:
            # for dag-former only
            x = self.gnn_model(x, batch.edge_index, edge_attr, batch.batch, batch.dag_rr_edge_index, batch.ptr,
                               pe=batch.pe, Lambda=batch.Lambda)
        elif hasattr(self.gnn_model, 'se'): # not a good way
            # for SAT only
            degree = batch.degree if hasattr(batch, 'degree') else None
            if 'pe' in batch:
                x = self.gnn_model(x, batch.edge_index, edge_attr, degree, batch.batch, batch.ptr,
                                   pe=batch.pe, Lambda=batch.Lambda)
            else:
                x = self.gnn_model(x, batch.edge_index, edge_attr, degree, batch.batch, batch.ptr)
            #x = self.gnn_model(x, batch.edge_index, edge_attr, degree, None, batch.ptr)
        else:
            # for other models
            x = self.gnn_model(x, batch.edge_index, edge_attr, batch.batch)

        # pooling
        x = global_mean_pool(x, batch.batch)

        return self.output_mlps(x)



