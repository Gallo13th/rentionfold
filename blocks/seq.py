import torch
from torch import nn
from torch.nn import functional as F
import math

class SeqAttwithPairBias(nn.Module):
    def __init__(self,seq_dim,pair_dim,n_heads,hidden_dim):
        super(SeqAttwithPairBias,self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ln_seq = nn.LayerNorm(seq_dim)
        self.ln_pair = nn.LayerNorm(pair_dim)
        self.W_b = nn.Linear(pair_dim,n_heads,bias=False)
        self.W_q = nn.Linear(seq_dim,n_heads*hidden_dim,bias=False)
        self.W_k = nn.Linear(seq_dim,n_heads*hidden_dim,bias=False)
        self.W_v = nn.Linear(seq_dim,n_heads*hidden_dim,bias=False)
        self.W_g = nn.Linear(seq_dim,n_heads*hidden_dim)
        self.W_o = nn.Linear(n_heads*hidden_dim,seq_dim)
        self.factor = 1/math.sqrt(hidden_dim)
    def forward(self,seq,pair):
        batch, length, seq_dim = seq.shape
        seq = self.ln_seq(seq)
        pair = self.ln_pair(pair)
        Q = self.W_q(seq)
        K = self.W_k(seq)
        V = self.W_v(seq)
        Q = Q.contiguous().view(batch,length,self.n_heads,self.hidden_dim)
        K = K.contiguous().view(batch,length,self.n_heads,self.hidden_dim)
        V = V.contiguous().view(batch,length,self.n_heads,self.hidden_dim)
        pair_bias = self.W_b(pair)
        pair_bias = pair_bias.permute(0,3,1,2)
        att_map = torch.einsum('binh,bjnh->bnij',Q,K)* self.factor + pair_bias
        att_map = F.softmax(att_map,dim=-1)
        gate = torch.sigmoid(self.W_g(seq))
        gate = gate.contiguous().view(batch,length,self.n_heads,self.hidden_dim)
        seq = torch.einsum('bnij,bjnh->binh',att_map,V)
        seq = seq * gate
        seq = seq.contiguous().view(batch,length,-1)
        seq = self.W_o(seq)
        return seq
class SeqAtt(nn.Module):
    def __init__(self,seq_dim,n_heads,hidden_dim):
        super(SeqAtt,self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ln_seq = nn.LayerNorm(seq_dim)
        self.W_q = nn.Linear(seq_dim,n_heads*hidden_dim,bias=False)
        self.W_k = nn.Linear(seq_dim,n_heads*hidden_dim,bias=False)
        self.W_v = nn.Linear(seq_dim,n_heads*hidden_dim,bias=False)
        self.W_g = nn.Linear(seq_dim,n_heads*hidden_dim)
        self.W_o = nn.Linear(n_heads*hidden_dim,seq_dim)
        self.factor = 1/math.sqrt(hidden_dim)
    def forward(self,seq):
        batch, length, seq_dim = seq.shape
        seq = self.ln_seq(seq)
        Q = self.W_q(seq)
        K = self.W_k(seq)
        V = self.W_v(seq)
        Q = Q.contiguous().view(batch,length,self.n_heads,self.hidden_dim)
        K = K.contiguous().view(batch,length,self.n_heads,self.hidden_dim)
        V = V.contiguous().view(batch,length,self.n_heads,self.hidden_dim)
        att_map = torch.einsum('binh,bjnh->bnij',Q,K)* self.factor
        att_map = F.softmax(att_map,dim=-1)
        gate = torch.sigmoid(self.W_g(seq))
        gate = gate.contiguous().view(batch,length,self.n_heads,self.hidden_dim)
        seq = torch.einsum('bnij,bjnh->binh',att_map,V)
        seq = seq * gate
        seq = seq.contiguous().view(batch,length,-1)
        seq = self.W_o(seq)
        return seq
class SeqTransition(nn.Module):
    def __init__(self, seq_dim):
        super(SeqTransition, self).__init__()
        self.ln = nn.LayerNorm(seq_dim)
        self.linear_1 = nn.Linear(seq_dim, 4*seq_dim)
        self.linear_2 = nn.Linear(4*seq_dim, 2*seq_dim)
        self.linear_3 = nn.Linear(2*seq_dim, seq_dim)
        self.relu = nn.ReLU()
    def forward(self, seq):
        seq = self.ln(seq)
        seq = self.linear_1(seq)
        seq = self.relu(seq)
        seq = self.linear_2(seq)
        seq = self.relu(seq)
        seq = self.linear_3(seq)
        return seq