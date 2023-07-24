import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
 
device = torch.device('cuda:0' )
 
class RowAttention(nn.Module):
   
    def __init__(self,embed_dim,n_heads):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(RowAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.mha_att = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=n_heads,batch_first=True)
       
    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,L,h,w)
        '''
       
        b, _, h, w = x.size()
       
        x = x.permute(0,2,1,3).contiguous().view(b*h, -1,w)  #size = (b*h,L,w)
        x,_ = self.mha_att(x,x,x)
        x = x.view(b,h,-1,w).permute(0,2,1,3)
 
        return x

class ColAttention(nn.Module):
    def __init__(self,embed_dim,n_heads):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(ColAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.mha_att = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=n_heads,batch_first=True)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,L,h,w)
        '''
 
        b, _, h, w = x.size()       
        x = x.permute(0,3,1,2).contiguous().view(b*w, -1,h)
        x,_ = self.mha_att(x,x,x)
        x = x.view(b,w,-1,h).permute(0,2,3,1)
        return x

class Stem(nn.Module):
    
    def __init__(self,c_dim) :
        '''
        Parameters
        ---------
        c_dim : int
        '''
        super(Stem,self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1,out_channels=c_dim,kernel_size=(1,1))

    def forward(self,x):
        '''
        x : (batch_size,1,length,embed_dim)
        output : (batch_size,c_dim,length,embed_dim)
        '''
        return self.conv2d(x)

class AxialAttetnion(nn.Module):
    
    def __init__(self,row_dim,col_dim,r_n_head,c_n_head):
        super(AxialAttetnion,self).__init__()
        self.rowattention = RowAttention(embed_dim=row_dim,n_heads=r_n_head)
        self.columnattention = ColAttention(embed_dim=col_dim,n_heads=c_n_head)
        
    def forward(self,x):
        r = self.rowattention(x)
        c = self.columnattention(x)
        return r+c

        
t = torch.randn(32,64,6)
t = t.resize(32,1,64,6)
t = Stem(128)(t)
t = t.permute(0,2,1,3)
t = AxialAttetnion(6,128,3,8)(t)
print(t.size())