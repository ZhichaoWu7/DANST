import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch


class FCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCNBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.LeakyReLU(0.01, inplace=True))    
        self._initialize_weights() 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.xavier_uniform_(m.weight) 
                init.kaiming_normal_(m.weight) 
                if m.bias is not None:
                    init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1) 
                init.constant_(m.bias, 0)   
    
    def forward(self, x):
        out = self.layer(x)
        return out
    

class VA_FCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VA_FCNBlock, self).__init__()
        self.mu = nn.Linear(in_features=in_dim,out_features=out_dim)
        self.log_sigma = nn.Linear(in_features=in_dim,out_features=out_dim)   
        self._initialize_weights() 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.xavier_uniform_(m.weight) 
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1) 
                init.constant_(m.bias, 0) 
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, x):
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        out = self.reparameterize(mu,log_sigma)
        return out

    
class GCNBlock(nn.Module): 
    def __init__(self, in_dim, out_dim, dropout=0.0, act=F.relu):
        super(GCNBlock, self).__init__()
        self.in_features = in_dim
        self.out_features = out_dim 
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.BN = nn.BatchNorm1d(self.out_features)
        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight)
        z = torch.mm(adj, z) 
        z = self.BN(z)
        emb = self.act(z) 
        return emb

    
    
class VA_GCNBlock(nn.Module): 
    def __init__(self, in_dim, out_dim):
        super(VA_GCNBlock, self).__init__()
        self.in_features = in_dim
        self.out_features = out_dim 

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2) 
        eps = torch.randn_like(std) 
        return mu + eps * std

    def forward(self, feat, adj):
        z1 = torch.mm(feat, self.weight1)
        mu = torch.mm(adj, z1)

        z2 = torch.mm(feat, self.weight2)
        log_sigma = torch.mm(adj, z2) 
        
        out=self.reparameterize(mu,log_sigma)
        return out
    
