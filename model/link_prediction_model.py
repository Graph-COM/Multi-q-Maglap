from torch import nn
import torch
from model.nns import MLPs
from utils.handle_complex import ComplexHandler, SparseComplexNetwork, SparseComplexNetworkSVD
from torch_geometric.utils import degree

class LinkPrediction(nn.Module):
    def __init__(self, num_node_types, node_emb_dim, pe_dim, eigval_dim, gnn_model, out_dim=1, pe_model=None):
        super(LinkPrediction, self).__init__()
        if num_node_types > 0:
            self.node_embedder = nn.Embedding(num_node_types, node_emb_dim)
        else:
            self.node_embedder = nn.Linear(2, node_emb_dim)
        self.num_node_types = num_node_types
        self.node_emb_dim = node_emb_dim
        if pe_model is None:
            self.pe_projection = nn.Linear(pe_dim + eigval_dim, node_emb_dim)
        else:
            self.pe_model = pe_model
        self.gnn_model = gnn_model
        self.output_mlp = MLPs(2 * gnn_model.out_dims, gnn_model.out_dims, out_dim, 8)

    def forward(self, batch):
        if self.num_node_types > 0:
            x = self.node_embedder(batch.x)
        else:
            #x = torch.zeros([batch.num_nodes, self.node_emb_dim]).to(batch.edge_index.device)
            x = self.node_embedder(batch.degree)
        # add positional encodings
        if hasattr(self, 'pe_projection'):
            if batch.pe.dtype == torch.complex64:
                z = torch.cat([batch.pe.flatten(1).real, batch.pe.flatten(1).imag, batch.Lambda.flatten(1)[batch.batch]], dim=-1)
            else:
                z = torch.cat([batch.pe.flatten(1), batch.Lambda.flatten(1)[batch.batch]], dim=-1) # concat eigenvalues
            z = self.pe_projection(z)
        elif hasattr(self, 'pe_model'):
            z = self.pe_model(batch.pe, batch.Lambda, batch.edge_index, batch.batch)
        else:
            raise Exception("pe preprocessing module not defined.")
        x = x + z
        # run gnn model
        x = self.gnn_model(x, batch.edge_index, None, batch.batch)
        # broadcast node representations to links
        x = torch.cat([x[batch.dist_index[0]], x[batch.dist_index[1]]], dim=-1)
        # get link prediction labels
        x = self.output_mlp(x)
        return x

class LinkPredictionInvariant(nn.Module):
    def __init__(self, num_node_types, node_emb_dim, pe_dim, q_dim, gnn_model, out_dim=1, pe_type='maglap',
                 handle_symmetry='spe'):
        super(LinkPredictionInvariant, self).__init__()
        if num_node_types > 0:
            self.node_embedder = nn.Embedding(num_node_types, node_emb_dim)
        else:
            self.node_embedder = nn.Linear(2, node_emb_dim)
        self.num_node_types = num_node_types
        self.node_emb_dim = node_emb_dim
        #self.pe_projection = nn.Linear(pe_dim + eigval_dim, gnn_model.out_dims)
        eigval_hidden_dim = 8
        #self.pe_projection = nn.Linear(2 * q_dim * eigval_hidden_dim, gnn_model.out_dims)
        self.eigval_encoder = MLPs(1, 32, eigval_hidden_dim, 3)
        self.gnn_model = gnn_model # this should be a basis-invariant gnn model
        link_dim = 3 * gnn_model.out_dims
        self.output_mlp = MLPs(link_dim, gnn_model.out_dims, out_dim, 8)
        complex_net_type = handle_symmetry
        #self.complex_handler = ComplexHandler(pe_dim=pe_dim, q_dim=q_dim, pe_type=pe_type)
        if pe_type == 'svd':
            self.sparse_complex_net = SparseComplexNetworkSVD(pe_dim=pe_dim, pe_type=pe_type,
                                                              out_dim=gnn_model.out_dims, network_type=complex_net_type)
        else:
            self.sparse_complex_net = SparseComplexNetwork(pe_dim=pe_dim, q_dim=q_dim, pe_type=pe_type,
                                                       out_dim=gnn_model.out_dims, network_type=complex_net_type)

    def forward(self, batch):
        if self.num_node_types > 0:
            x = self.node_embedder(batch.x)
        else:
            # TO DO: add magnitude of pe as initial node features
            #x = torch.zeros([batch.num_nodes, self.node_emb_dim]).to(batch.edge_index.device)
            x = self.node_embedder(batch.degree)


        z = batch.pe
        # run an equivariant gnn model: x is scalar and z is vector (w.r.t. basis transformation)
        x, z = self.gnn_model(x, z, batch.Lambda, batch.edge_index, None, batch.batch)

        # equivariant -> invariant readout
        #z = self.complex_handler.merge_real_imag(z) # [BN, Q, pe_dim]
        #Lambda = batch.Lambda.unflatten(-1, (self.complex_handler.q_dim, -1))
        #Lambda = self.eigval_encoder(Lambda.unsqueeze(-1))
        #Lambda = Lambda[batch.batch[batch.dist_index[0]]] # [BN, Q, pe_dim]
        #temp = Lambda * z[batch.dist_index[0]].unsqueeze(-1) * torch.conj(z[batch.dist_index[1]]).unsqueeze(-1)
        #temp = temp.sum(2).flatten(1)
        #temp = torch.cat([temp.real, temp.imag], dim=-1)
        #temp = self.pe_projection(temp)
        temp = self.sparse_complex_net(z, batch.Lambda, batch.dist_index, batch.batch)


        # merge invariant and equivariant features
        #z = torch.cat([z, batch.Lambda[batch.batch]], dim=-1)
        #z = self.pe_projection(z)
        #x = x + 0*z


        # broadcast node representations to links
        x = torch.cat([x[batch.dist_index[0]], x[batch.dist_index[1]]], dim=-1)
        x = torch.cat([x, temp], dim=-1)
        # get link prediction labels
        x = self.output_mlp(x)
        return x
