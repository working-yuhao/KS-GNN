import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, ChebConv  



class MLP(torch.nn.Module):
    def __init__(self, input_dim,  hidden_dim, output_dim, feature_dim = None,
                 feature_pre=False, layer_num=2, dropout=False, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        # x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        return x



class KSNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=2, conv_num = 5, alpha=0.9):
        super(KSNN, self).__init__(aggr='max')
        
        self.input_dim = input_dim
        self.encoder = MLP(self.input_dim, hidden_dim, output_dim, layer_num = layer_num)
        self.decoder = MLP(output_dim, hidden_dim, self.input_dim, layer_num = layer_num)     
        self.alpha = alpha
        self.conv_num = conv_num
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.pca_Z = None
        self.pca_v = None
        
    def msg(self, x, edge_index):
        de_X = self.decoder(x)
        x_j = self.propagate(edge_index, x=de_X)*self.alpha
        tmp_x = torch.max(de_X,x_j) 
        return self.encoder(tmp_x)

    def kspca_msg(self, x, edge_index):
        de_X = x @ self.pca_v.t()
        x_j = self.propagate(edge_index, x=de_X)*self.alpha
        tmp_x = torch.max(de_X,x_j) 
        return tmp_x @ self.pca_v

    def kspca(self, X, edge_index):
        x = self.pca_Z if not self.pca_Z is None else self._set_pcaX(X)
        for i in range(self.conv_num):
            x = self.kspca_msg(x,edge_index)
        return x

    def forward(self, x, edge_index):
        x = self.encoder(x)
        for i in range(self.conv_num):
            x = self.msg(x, edge_index)
        return x

    def sage_msg(self, x, edge_index):
        de_X = x
        x_j = self.propagate(edge_index, x=de_X)
        return torch.max(de_X,x_j) 

    def sage_forward(self, x, edge_index):
        x = self.encoder(x)
        for i in range(self.conv_num):
            x = self.sage_msg(x, edge_index)
        return x

    def _set_pcaX(self, X):
        u,s,v = torch.svd(X)
        self.pca_v = v[:,:self.output_dim]
        self.pca_Z = X @ self.pca_v
        return self.pca_Z

    def pca_msg(self, x, edge_index):
        x_j = self.propagate(edge_index, x=x)*self.alpha
        return torch.max(x,x_j) 
    
    def pca(self, X, edge_index):
        x = self.pca_Z if not self.pca_Z is None else self._set_pcaX(X)
        for i in range(self.conv_num):
            x = self.pca_msg(x,edge_index)
        return x
    

