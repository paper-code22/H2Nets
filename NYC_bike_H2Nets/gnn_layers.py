import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
import torch_geometric.nn as g_nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
import math
GNN_ATT=False
GAT_HEAD=1

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv,stdv)
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)        
        

class BiGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggr='add', bias=True, mode='cat',**kwargs):
        super(BiGraphConv,self).__init__()
        self.mode=mode
        if GNN_ATT:
            self.v2e = NeiConv(in_channels, in_channels, aggr=aggr)
            self.e2v = AttConv(in_channels, out_channels, aggr=aggr, heads=GAT_HEAD, flow='target_to_source')
        else:
            self.v2e = NeiConv(in_channels, out_channels, aggr=aggr)
            self.e2v = NeiConv(out_channels, out_channels, aggr=aggr, flow='target_to_source')
        if mode == 'cat':
            self.lin = nn.Linear(in_channels+out_channels,out_channels, bias=bias)
        elif mode == 'sum':
            self.lin = nn.Linear(in_channels, out_channels, bias=bias)
    def forward(self, x, edge_index, edge_weight=None, size=None, ret_rec=False,att=False):
        x_v = x
        x_e = self.v2e(x_v, edge_index, edge_weight=edge_weight, size=size)
        
        if GNN_ATT:
            if att:
                x_v,alpha,nf= self.e2v(x_e,x_v,edge_index,size=size,att=True)
            else:
                x_v = self.e2v(x_e,x_v,edge_index,size=size)
        else:
            x_v = self.e2v(x_e, edge_index, edge_weight=edge_weight, size=[size[1], size[0]]) # reshape size
        if self.mode == 'cat':
            x = torch.cat([x,x_v],dim=-1)
            x = self.lin(x)
            if att:
                return x,alpha,nf
            return x
        elif self.mode == 'sum':
            x = self.lin(x)
            x = x_v+x
            if ret_rec:
                return x, x_e
            else:
                return x

class NeiConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(NeiConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        h = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)


    def message(self, h_j, edge_weight):

        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class AttConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=False, negative_slope=0.2, dropout=0.5, bias=False,aggr='mean', **kwargs):
        super(AttConv, self).__init__(aggr=aggr,**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.heads = heads
        self.negative_slope=negative_slope
        self.dropout=dropout

        self.v=None
        self.e_weight = Parameter(torch.Tensor(in_channels, heads*out_channels))
        self.v_weight = Parameter(torch.Tensor(in_channels, heads*out_channels))
        self.att = Parameter(torch.Tensor(1,1,heads,2*out_channels))
        self.concat=concat
        if bias and concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        glorot(self.e_weight)
        glorot(self.v_weight)
        glorot(self.att)
        zeros(self.bias)
    def forward(self,x_e, x_v,edge_index,size=None,att=False):
        x = torch.matmul(x_e,self.e_weight)
        self.v = torch.matmul(x_v, self.v_weight)
        if att:
            return self.propagate(edge_index,size=size,x=x), self.alpha, self.f
        return self.propagate(edge_index,size=size,x=x)
    def message(self, x_j, edge_index_i, size_i):
        x_j = x_j.view(x_j.shape[0],x_j.shape[1], self.heads, self.out_channels)
        x_i = self.v[edge_index_i]
        x_i = x_i.view(x_i.shape[0],x_i.shape[1],self.heads,self.out_channels)
        alpha = (torch.cat([x_i,x_j],dim=-1)*self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha,self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha = alpha.unsqueeze(-1)
        self.alpha=alpha
        self.f=edge_index_i
        return x_j*alpha
    def update(self,aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads*self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=-2)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class HigherOrderConv(nn.Module):
    """
    Higher order structure(s) convolution operation
    """

    def __init__(self, in_features, out_features, bias=True):
        super(HigherOrderConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.einsum('bni,io->bno', input, self.weight)
        output = adj.matmul(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

'''
np.random.seed(1)
in_channels, out_channels = (16, 32)
hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 3, 2, 1, 2],
                                [0, 1, 2, 1, 2, 1, 0, 3, 3, 4, 4]])
hyperedge_weight = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.7])
num_nodes = hyperedge_index[0].max().item() + 1
batch_size = 32
x = torch.randn((batch_size, num_nodes, in_channels))
enc = BiGraphConv(in_channels, out_channels, aggr='mean', mode='cat')
out = enc(x, hyperedge_index, size = [4, 5])
print(out.size())
'''