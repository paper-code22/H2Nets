import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from hyperconv import HypergraphConv
from gnn_layers import BiGraphConv, HigherOrderConv

class HoHyGCNGCN(nn.Module):
    def __init__(self, dim_in, dim_e_in, dim_out, window_len, link_len, embed_dim):
        super(HoHyGCNGCN, self).__init__()
        self.link_len = link_len
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in + 55, int(dim_out/2)))
        #if (dim_in-2)%16 ==0:
        #    self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, link_len, 1, int(dim_out / 4)))
        #else:
        #    self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, link_len, int(dim_in/2), int(dim_out / 4)))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        #self.T = nn.Parameter(torch.FloatTensor(window_len))
        self.enc_node = BiGraphConv(dim_in, int(dim_out / 4), aggr='mean', mode='cat')
        self.enc_edge = BiGraphConv(dim_e_in, int(dim_out / 8), aggr='mean', mode='cat')
        self.hoc = HigherOrderConv(dim_e_in, int(dim_out/8))

    def forward(self, x, x_e, x_time, x_window, node_embeddings, hodge_laplacian, incidence_matrix, hyper_edge_data, hyper_node_data):

        #S1: Laplacian construction
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        #S2: Laplacianlink
        for k in range(2, self.link_len):
            support_set.append(torch.mm(supports, support_set[k-1]))
        supports = torch.stack(support_set, dim=0)

        #S3: concatenate x and x_time
        x_time_variables = x_time.repeat(1, x.size(1), 1) #B, N, 55
        merge_x = torch.cat([x, x_time_variables], dim=-1) #B, N, 55 + dim_in

        #S4: spatial graph convolution
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) #N, link_len, dim_in + 55, dim_out/?
        bias = torch.matmul(node_embeddings, self.bias_pool) #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, merge_x) #B, link_len, N, dim_in + 55
        x_g = x_g.permute(0, 2, 1, 3) #B, N, link_len, dim_in + 55
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) #B, N, dim_out/?

        #S5: temporal graph convolution
        #weights_window = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_window) #N, link_len, dim_in, dim_out/?
        #x_w1 = torch.einsum("knm,btmi->btkni",supports, x_window) #B, T, link_len, N, dim_in
        #x_w1 = x_w1.permute(0,1,3,2,4) #B, T, N, link_len, dim_in
        #x_w = torch.einsum('btnki,nkio->btno', x_w1, weights_window) #B, T, N, dim_out/?
        #x_w = x_w.permute(0, 2, 3, 1) #B, N, dim_out/?, T
        #x_wconv = torch.matmul(x_w, self.T) #B, N, dim_out/?

        #S6: node: hyper representation learning
        x_node_hyper = F.dropout(x, p=0.5, training=self.training)
        x_node_hyper = self.enc_node(x_node_hyper, hyper_edge_data, size = [200, 39]) #B, N, dim_out/?

        #S7: edge: hyper representation learning
        x_edge_hyper = F.dropout(x_e, p=0.5, training=self.training) #B, M, 1
        #utilize hypernode information for edge learning, i.e., 100 edges and 26 hypernodes
        x_edge_hyper = self.enc_edge(x_edge_hyper, hyper_node_data, size=[100, 23])  #B, M, dim_out/?

        #S8: hodge Laplacian convolution operation
        x_e_hconv = self.hoc(x_edge_hyper, hodge_laplacian) #B, M, dim_out/?

        #S9: concatenate x_edge_hyper and x_e_hodge
        x_e_hh = torch.cat([x_edge_hyper, x_e_hconv], dim = -1) #B, M, dim_out/?
        x_e_hh = x_e_hh.permute(0, 2, 1) #B, dim_out/?, M

        #S10: edge to node aggregation
        x_edge_hh_node = torch.einsum('bom, mn->bon', x_e_hh, incidence_matrix.transpose(0,1)) #B, dim_out/?, N
        x_edge_hh_node = x_edge_hh_node.permute(0, 2, 1) #B, N, dim_out/?

        #S11: combination operation
        x_gwconv = torch.cat([x_gconv, x_node_hyper, x_edge_hh_node], dim = -1) + bias #B, N, dim_out
        return x_gwconv

