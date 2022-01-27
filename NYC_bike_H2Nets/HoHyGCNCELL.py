import torch
import torch.nn as nn
import torch.nn.functional as F
from HoHyGCN import HoHyGCNGCN

class HoHyGCRNNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_e_in, dim_out, window_len, link_len, embed_dim):
        super(HoHyGCRNNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = HoHyGCNGCN(dim_in+self.hidden_dim, dim_e_in, 2*dim_out, window_len, link_len, embed_dim)
        self.update = HoHyGCNGCN(dim_in+self.hidden_dim, dim_e_in, dim_out, window_len, link_len, embed_dim)

    def forward(self, x, state, x_full, node_embeddings, x_time, x_e, hodge_laplacian, incidence_matrix, hyper_edge_data, hyper_node_data):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1) #x + state
        z_r = torch.sigmoid(self.gate(input_and_state, x_e, x_time, x_full, node_embeddings, hodge_laplacian, incidence_matrix, hyper_edge_data, hyper_node_data))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, x_e, x_time, x_full, node_embeddings, hodge_laplacian, incidence_matrix, hyper_edge_data, hyper_node_data))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)