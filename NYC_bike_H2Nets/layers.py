import torch
import torch.nn as nn

class ID(nn.Module):
    def __init__(self):
        super(ID,self).__init__()
    def forward(self,x):
        return x
class FC(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, dropout=0, batchnorm=False):
        super(FC,self).__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_dim))
        
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
            
        if not activation:
            pass
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'elu':
            layers.append(nn.ELU())
        
        self.fc = nn.Sequential(*layers)
    def forward(self, x):
        in_size=x.size()
        if len(in_size) > 2:
            x = x.view(-1, x.size(-1))
        x = self.fc(x)
        if len(in_size) > 2:
            x = x.view(list(in_size[:-1])+[-1])
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=3, activation=None, dropout=0, batchnorm=False):
        super(MLP, self).__init__()
        layers = []
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim]*(num_layers-1)
        num_layers = len(hidden_dim)+1
        
        layers.append(FC(in_dim, hidden_dim[0], activation=activation, dropout=dropout, batchnorm=batchnorm))
        for i in range(num_layers-2):
            layers.append(FC(hidden_dim[i], hidden_dim[i+1], activation=activation, dropout=dropout, batchnorm=batchnorm))
        layers.append(FC(hidden_dim[-1], out_dim, activation=activation, dropout=False, batchnorm=False))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.mlp(x)
        
