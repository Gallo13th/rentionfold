import torch
from torch import nn


class ConstraintHead(nn.Module):
    
    def __init__(self,pair_dim):
        super(ConstraintHead,self).__init__()
        self.map_predict_dist_P = nn.Linear(in_features=pair_dim,out_features=40)
        self.map_predict_dist_C4 = nn.Linear(in_features=pair_dim,out_features=40)
        self.map_predict_dist_N = nn.Linear(in_features=pair_dim,out_features=40)
        self.map_predict_Lambda = nn.Linear(in_features=pair_dim,out_features=25)
        self.map_predict_Omega = nn.Linear(in_features=pair_dim,out_features=25)
        
    def forward(self,pair):
        dist_C4 = self.map_predict_dist_C4(pair).permute(0,3,1,2)
        dist_C4 = torch.softmax(dist_C4+dist_C4.permute(0,1,3,2),dim=1)
        
        dist_P = self.map_predict_dist_P(pair).permute(0,3,1,2)
        dist_P = torch.softmax(dist_P+dist_P.permute(0,1,3,2),dim=1)
        
        dist_N = self.map_predict_dist_N(pair).permute(0,3,1,2)
        dist_N = torch.softmax(dist_N+dist_N.permute(0,1,3,2),dim=1)
        
        Omega = self.map_predict_Omega(pair).permute(0,3,1,2)
        Omega = torch.softmax(Omega+Omega.permute(0,1,3,2),dim=1)
        
        Lambda = self.map_predict_Lambda(pair).permute(0,3,1,2)
        Lambda = torch.softmax(Lambda,dim=1)
        
        return dist_C4,dist_P,dist_N,Omega,Lambda
    
class BackboneHead(nn.Module):
    
    def __init__(self,seq_dim):
        super(BackboneHead,self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(in_features=32,out_features=32),
             nn.ReLU(),
             nn.Linear(in_features=32,out_features=32),
             nn.ReLU()
             )
        self.l_in = nn.Linear(in_features=seq_dim,out_features=32)
        self.l_out = nn.Linear(in_features=32,out_features=8*24)
        
    def forward(self,seq):
        b,l,_ = seq.shape
        seq = self.l_in(seq)
        for _ in range(2):
            seq = self.trans(seq)
        seq = self.l_out(seq)
        seq = seq.view(b,l,24,8)
        seq = torch.softmax(seq,dim=-1)
        return seq