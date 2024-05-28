 


import torch
from torch import nn
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential, Dropout

from torch_geometric.nn import GINEConv, GCNConv, GPSConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.attention import PerformerAttention

from torch_geometric_signed_directed.nn import MSConv
from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer

from models.base_model import MLPs
from models.middle_model import MiddleModel


class NodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_embedding_list = torch.nn.ModuleList()
        feature_dim_list = [16, 32, 128]
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
    

class AMPModel(nn.Module):
    def __init__(self, args):
        super(AMPModel, self).__init__()
        self.args = args
        self.pe_type = args.get('pe_type')
        self.hidden_dim = args['hidden_dim']


        self.node_encoder = NodeEncoder(self.hidden_dim // 3)

        # define middle model
        self.middle_model1 = MiddleModel(self.args)
        self.bridge_layer = nn.Linear(self.middle_model1.out_dims, self.hidden_dim) # align dimension
        self.middle_model2 = MiddleModel(self.args)

        # define final layer
        self.gain_mlp = MLPs(self.middle_model2.out_dims, self.middle_model2.out_dims, 1, args['mlp_out']['num_layer'])
        self.pm_mlp = MLPs(self.middle_model2.out_dims, self.middle_model2.out_dims, 1, args['mlp_out']['num_layer'])
        self.bw_mlp = MLPs(self.middle_model2.out_dims, self.middle_model2.out_dims, 1, args['mlp_out']['num_layer'])

    def forward(self, batch_data):
        #pre-process the node, edge data
        x = self.node_encoder(batch_data.x)
        # in the first layer, call the middle model to only message-passing through sub edge indices
        if self.pe_type is None:
            x = self.middle_model1(x, batch_data.sub_edge_index, batch_data.batch)
        else:
            x = self.middle_model1(x, batch_data.sub_edge_index, batch_data.batch, 
                                   mag_pe = getattr(batch_data, 'mag_pe', None), lap_pe = getattr(batch_data, 'lap_pe', None),
                                  Lambda = batch_data.Lambda)

        # align dimension between middle model 1 and 2
        x = self.bridge_layer(x)
        # in the 2nd layer, call another middle model to do message passing through the entire net edge indices
        if self.pe_type is None:
            x = self.middle_model2(x, batch_data.edge_index, batch_data.batch)
        else:
            x = self.middle_model2(x, batch_data.edge_index, batch_data.batch, 
                                  mag_pe = getattr(batch_data, 'mag_pe', None), lap_pe = getattr(batch_data, 'lap_pe', None),
                                  Lambda = batch_data.Lambda)
        #final MLP
        x_add = global_add_pool(x, batch_data.batch)
        #x_max = global_max_pool(x, batch_data.batch)
        gain = self.gain_mlp(x_add)
        pm = self.pm_mlp(x_add)
        bw = self.bw_mlp(x_add)
        return gain, pm, bw