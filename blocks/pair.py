# Author: Robin Pearce
# To report bugs and questions email: robpearc@umich.edu 

import torch
from torch import nn
from torch.nn import functional as F
import math


class Triangle_Att_Start(nn.Module):
    def __init__(self,pair_dim,n_heads,hidden_dim):
        super(Triangle_Att_Start,self).__init__()

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(pair_dim)
        self.W_q = nn.Linear(pair_dim,hidden_dim*n_heads,bias=False)
        self.W_k = nn.Linear(pair_dim,hidden_dim*n_heads,bias=False)
        self.W_v = nn.Linear(pair_dim,hidden_dim*n_heads,bias=False)
        self.W_b = nn.Linear(pair_dim,n_heads,bias=False)
        self.W_g = nn.Linear(pair_dim,hidden_dim*n_heads)
        self.W_o = nn.Linear(hidden_dim*n_heads,pair_dim)
        
        self.factor = 1/math.sqrt(hidden_dim)

    def forward(self,pair):
        batch,length, _, pair_dim = pair.shape
        
        pair = self.ln(pair)
        
        Q = self.W_q(pair)
        K = self.W_k(pair)
        V = self.W_v(pair)

        Q = Q.contiguous().view(batch,length,length,self.n_heads,self.hidden_dim)
        K = K.contiguous().view(batch,length,length,self.n_heads,self.hidden_dim)
        V = V.contiguous().view(batch,length,length,self.n_heads,self.hidden_dim)

        bias = self.W_b(pair)
        bias = bias.permute(0,3,2,1).contiguous().view(batch,self.n_heads,1,length,length)
        attn_map = torch.einsum("bijnh,biknh->bnijk", Q, K) * self.factor + bias
        attn_map = F.softmax(attn_map,dim=-1)

        gate = torch.sigmoid(self.W_g(pair))
        gate = gate.contiguous().view(batch,length,length,self.n_heads,self.hidden_dim)

        pair = torch.einsum('bnijk,biknh->bijnh',attn_map, V)
        pair = pair * gate
        pair = pair.contiguous().view(batch,length,length,-1)
        pair = self.W_o(pair)

        return pair

class Triangle_Att_End(nn.Module):
    def __init__(self,pair_dim,n_heads,hidden_dim):
        super(Triangle_Att_End,self).__init__()
        
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(pair_dim)
        self.W_q = nn.Linear(pair_dim,hidden_dim*n_heads,bias=False)
        self.W_k = nn.Linear(pair_dim,hidden_dim*n_heads,bias=False)
        self.W_v = nn.Linear(pair_dim,hidden_dim*n_heads,bias=False)
        self.W_b = nn.Linear(pair_dim,n_heads,bias=False)
        self.W_g = nn.Linear(pair_dim,hidden_dim*n_heads)
        self.W_o = nn.Linear(hidden_dim*n_heads,pair_dim)
        
        self.factor = 1/math.sqrt(hidden_dim)

    def forward(self,pair):
        batch,length, _, pair_dim = pair.shape
        
        pair = self.ln(pair)
        
        Q = self.W_q(pair)
        K = self.W_k(pair)
        V = self.W_v(pair)

        Q = Q.contiguous().view(batch,length,length,self.n_heads,self.hidden_dim)
        K = K.contiguous().view(batch,length,length,self.n_heads,self.hidden_dim)
        V = V.contiguous().view(batch,length,length,self.n_heads,self.hidden_dim)

        bias = self.W_b(pair)
        bias = bias.permute(0,3,1,2).contiguous().view(batch,self.n_heads,1,length,length)
        attn_map = torch.einsum("bijnh,bkinh->bnijk", Q, K) * self.factor + bias
        attn_map = F.softmax(attn_map,dim=-1)

        gate = torch.sigmoid(self.W_g(pair))
        gate = gate.contiguous().view(batch,length,length,self.n_heads,self.hidden_dim)

        pair = torch.einsum('bnijk,bkjnh->bijnh',attn_map, V)
        pair = pair * gate
        pair = pair.contiguous().view(batch,length,length,-1)
        pair = self.W_o(pair)

        return pair

class Triangle_Update_Outgoing(nn.Module):
    def __init__(self,pair_dim,hidden_dim):
        super(Triangle_Update_Outgoing,self).__init__()

        self.ln = nn.LayerNorm(pair_dim)
        self.ln_hidden = nn.LayerNorm(hidden_dim)
        self.W_a = nn.Linear(pair_dim,hidden_dim)
        self.gate_a = nn.Linear(pair_dim,hidden_dim)
        self.W_b = nn.Linear(pair_dim,hidden_dim)
        self.gate_b = nn.Linear(pair_dim,hidden_dim)
        self.W_o = nn.Linear(hidden_dim,pair_dim)
        self.W_g = nn.Linear(pair_dim,pair_dim)

    def forward(self,pair):
        pair = self.ln(pair)
        
        a = self.W_a(pair) 
        g_a = torch.sigmoid(self.gate_a(pair))
        a = a * g_a

        b = self.W_b(pair)
        g_b = torch.sigmoid(self.gate_b(pair))
        b = b * g_b

        g_o = torch.sigmoid(self.W_g(pair))
        
        pair = torch.einsum('bikh,bjkh->bijh',a,b)
        pair = self.ln_hidden(pair)
        pair = self.W_o(pair)
        pair = pair * g_o
        
        return pair

class Triangle_Update_Incoming(nn.Module):
    def __init__(self,pair_dim,hidden_dim):
        super(Triangle_Update_Incoming,self).__init__()

        self.ln = nn.LayerNorm(pair_dim)
        self.ln_hidden = nn.LayerNorm(hidden_dim)
        self.W_a = nn.Linear(pair_dim,hidden_dim)
        self.gate_a = nn.Linear(pair_dim,hidden_dim)
        self.W_b = nn.Linear(pair_dim,hidden_dim)
        self.gate_b = nn.Linear(pair_dim,hidden_dim)
        self.W_o = nn.Linear(hidden_dim,pair_dim)
        self.W_g = nn.Linear(pair_dim,pair_dim)

    def forward(self,pair):
        pair = self.ln(pair)
        
        a = self.W_a(pair) 
        g_a = torch.sigmoid(self.gate_a(pair))
        a = a * g_a

        b = self.W_b(pair)
        g_b = torch.sigmoid(self.gate_b(pair))
        b = b * g_b

        g_o = torch.sigmoid(self.W_g(pair))
        
        pair = torch.einsum('bijh,bikh->bjkh',a,b)
        pair = self.ln_hidden(pair)
        pair = self.W_o(pair)
        pair = pair * g_o
        
        return pair

class Pair_Transition(nn.Module):
    def __init__(self,pair_dim,hidden_dim):
        super(Pair_Transition,self).__init__()
        
        self.ln = nn.LayerNorm(pair_dim)
        self.l1 = nn.Linear(pair_dim,pair_dim*hidden_dim)
        self.l2 = nn.Linear(pair_dim*hidden_dim,pair_dim)
        
        self.relu = nn.ReLU()

    def forward(self,pair):
        pair = self.ln(pair)
        pair = self.l1(pair)
        pair = self.relu(pair)
        pair = self.l2(pair)
        
        return pair