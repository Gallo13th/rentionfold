import torch
from torch import nn
from torch.nn import functional as F
import math

class MSA_Row_Att(nn.Module):
    def __init__(self,msa_dim,pair_dim,n_heads,hidden_dim):
        super(MSA_Row_Att,self).__init__()
        self.msa_dim = msa_dim
        self.pair_dim = pair_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        
        self.ln_msa = nn.LayerNorm(self.msa_dim)
        self.ln_pair = nn.LayerNorm(self.pair_dim)
        self.W_b = nn.Linear(self.pair_dim,self.n_heads,bias=False)
        self.W_q = nn.Linear(self.msa_dim,self.n_heads*self.hidden_dim,bias=False)
        self.W_k = nn.Linear(self.msa_dim,self.n_heads*self.hidden_dim,bias=False)
        self.W_v = nn.Linear(self.msa_dim,self.n_heads*self.hidden_dim,bias=False)
        self.W_g = nn.Linear(self.msa_dim,self.n_heads*self.hidden_dim)
        self.W_o = nn.Linear(self.n_heads*self.hidden_dim,self.msa_dim)
        
        self.factor = 1/math.sqrt(self.hidden_dim)
        
    def forward(self,msa,pair):
        batch,num_seqs,length,embed_dim = msa.shape
        
        msa = self.ln_msa(msa)
        pair = self.ln_pair(pair)

        Q = self.W_q(msa) 
        K = self.W_k(msa)
        V = self.W_v(msa)
        
        Q = Q.contiguous().view(batch,num_seqs,length,self.n_heads,self.hidden_dim)
        K = K.contiguous().view(batch,num_seqs,length,self.n_heads,self.hidden_dim)
        V = V.contiguous().view(batch,num_seqs,length,self.n_heads,self.hidden_dim)

        Q = Q * self.factor

        pair_bias = self.W_b(pair)
        pair_bias = pair_bias.permute(0,3,1,2)
        pair_bias = pair_bias.contiguous().view(batch,num_seqs,-1,length,length)
        att_map = torch.einsum('brinh,brjnh->brnij',Q,K) + pair_bias
        att_map = F.softmax(att_map,dim=-1)
        
        gate = torch.sigmoid(self.W_g(msa))
        gate = gate.contiguous().view(batch,num_seqs,length,self.n_heads,self.hidden_dim)
        
        out = torch.einsum('brnij,brjnh->brinh',att_map,V)
        out = out * gate
        out = out.contiguous().view(batch,num_seqs,length,-1)
        msa_embed_out = self.W_o(out)
        
        return msa_embed_out

class MSA_Col_Att(nn.Module):
    
    def __init__(self,msa_dim,n_heads,hidden_dim):
        super(MSA_Col_Att,self).__init__()
        self.msa_dim = msa_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        
        self.ln_msa = nn.LayerNorm(self.msa_dim)

        self.W_q = nn.Linear(self.msa_dim,self.n_heads*self.hidden_dim,bias=False)
        self.W_k = nn.Linear(self.msa_dim,self.n_heads*self.hidden_dim,bias=False)
        self.W_v = nn.Linear(self.msa_dim,self.n_heads*self.hidden_dim,bias=False)
        self.W_g = nn.Linear(self.msa_dim,self.n_heads*self.hidden_dim)
        self.W_o = nn.Linear(self.n_heads*self.hidden_dim,self.msa_dim)
        
        self.factor = 1/math.sqrt(self.hidden_dim)

    def forward(self,msa):
        batch,num_seqs,length,embed_dim = msa.shape
        
        msa = self.ln_msa(msa)

        Q = self.W_q(msa) 
        K = self.W_k(msa)
        V = self.W_v(msa)
        
        Q = Q.contiguous().view(batch,num_seqs,length,self.n_heads,self.hidden_dim)
        K = K.contiguous().view(batch,num_seqs,length,self.n_heads,self.hidden_dim)
        V = V.contiguous().view(batch,num_seqs,length,self.n_heads,self.hidden_dim)

        Q = Q * self.factor
        
        attn_map = torch.einsum("bicnh,bjcnh->bncij", Q, K)
        attn_map = F.softmax(attn_map,dim=-1)

        gate = torch.sigmoid(self.W_g(msa))
        gate = gate.contiguous().view(batch,num_seqs,length,self.n_heads,self.hidden_dim)
        
        out = torch.einsum('bncij,bjcnh->bicnh',attn_map, V) 
        out = out * gate
        out = out.contiguous().view(batch,num_seqs,length,-1)
        m_embed_out = self.W_o(out)
        
        return m_embed_out

class MSA_Transition(nn.Module):
    def __init__(self,msa_dim,hidden_dim):
        super(MSA_Transition,self).__init__()
        self.msa_dim = msa_dim
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(self.msa_dim)
        self.linear_trans_1 = nn.Linear(self.msa_dim,self.msa_dim*self.hidden_dim)
        self.linear_trans_2 = nn.Linear(self.msa_dim*self.hidden_dim,self.msa_dim)

        self.relu = nn.ReLU()

    def forward(self,msa):
        msa = self.ln(msa)
        msa = self.linear_trans_1(msa)
        msa = self.relu(msa)
        msa = self.linear_trans_2(msa)
        
        return msa

class MSA_Outer_Product_Mean(nn.Module):
    def __init__(self,msa_dim,pair_dim,hidden_dim):
        super(MSA_Outer_Product_Mean,self).__init__()
        self.msa_dim = msa_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(self.msa_dim)
        self.l1 = nn.Linear(self.msa_dim,self.hidden_dim)
        self.l2 = nn.Linear(self.msa_dim,self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim*self.hidden_dim,self.pair_dim)

    def forward(self,msa):
        batch,num_seqs, length, embed_dim = msa.shape
        
        msa = self.ln(msa)

        msa_1 = self.l1(msa)
        msa_2 = self.l2(msa)
        pair_opm = torch.einsum('brix,brjy->brijxy',msa_2,msa_1)
        pair_opm = torch.mean(pair_opm, dim=1)
        pair_opm = pair_opm.contiguous().view(batch,length,length,-1)
        pair_opm = self.l3(pair_opm)
        
        return pair_opm