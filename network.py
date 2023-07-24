import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SeqEmbed(nn.Module):
    '''
    Parameters:
        seq_dim: int, dimension of input sequence embedding
        seq_embed_dim: int, dimension of output sequence embedding
        pair_embed_dim: int, dimension of pair embedding
        device: torch.device, device to run the model
        n_pos_bins_seq: int, number of bins for relative position embedding of sequence
        n_pos_bins_pair: int, number of bins for relative position embedding of pair

    Inputs:
        seq: torch.Tensor, shape [batch_size,seq_len,seq_dim], sequence embedding
    
    Outputs:
        seq_embed: torch.Tensor, shape [batch_size,seq_len,seq_embed_dim], sequence embedding
        pair_embed: torch.Tensor, shape [batch_size,seq_len,seq_len,pair_embed_dim], pair embedding
    '''

    def __init__(self,seq_dim,seq_embed_dim,pair_embed_dim,device,n_pos_bins_seq,n_pos_bins_pair):

        super(SeqEmbed,self).__init__()
        self.seq_dim = seq_dim
        self.pair_dim = seq_dim ** 2
        self.seq_embed_dim = seq_embed_dim
        self.pair_embed_dim = pair_embed_dim
        self.device = device

        self.W_seq = nn.Linear(self.seq_dim,self.seq_embed_dim)
        self.W_pair = nn.Linear(self.pair_dim,self.pair_embed_dim)

        self.pos_seq = self.pos_embed_seq(device,n_pos_bins_seq)
        self.pos_pair = self.pos_embed_pair(device,n_pos_bins_pair)

        self.pos_seq_embed = nn.Linear(n_pos_bins_seq,self.seq_embed_dim)
        self.pos_pair_embed = nn.Linear(n_pos_bins_pair,self.pair_embed_dim)

        self.seq_pair_i = torch.nn.Linear(self.seq_dim,self.pair_embed_dim)
        self.seq_pair_j = torch.nn.Linear(self.seq_dim,self.pair_embed_dim)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.W_seq.weight)
        nn.init.xavier_normal_(self.W_pair.weight)
        nn.init.xavier_normal_(self.pos_seq_embed.weight)
        nn.init.xavier_normal_(self.pos_pair_embed.weight)
        nn.init.xavier_normal_(self.seq_pair_i.weight)
        nn.init.xavier_normal_(self.seq_pair_j.weight)

    def seq2pair(self,seq):
        b,l,_ = seq.shape
        pair = torch.einsum('bim,bjn->bijmn',seq,seq)
        return pair

    def pos_embed_seq(self,device,n_pos_bins):
        pos = torch.arange(512)
        rel_pos = ((pos[:,None] & (1 << np.arange(n_pos_bins)))) > 0
        return rel_pos.float().to(device)
    
    def pos_embed_pair(self,device,n_pos_bins):
        pos = torch.arange(512)
        rel_pos = pos[None,:]-pos[:,None]
        rel_pos = rel_pos.clamp(-32,32)
        rel_pos_encode = torch.nn.functional.one_hot(rel_pos+32,n_pos_bins)
        return rel_pos_encode.float().to(device)

    def forward(self,seq):
        b,length,_ = seq.shape

        seq_embed = self.W_seq(seq)
        pair = self.seq2pair(seq).view(b,length,length,-1)
        pair_embed = self.W_pair(pair)

        pos_embed = self.pos_seq_embed(self.pos_seq[:length])
        seq_embed = seq_embed + pos_embed # [batch_size,seq_len,seq_embed_dim]

        pos_embed = self.pos_pair_embed(self.pos_pair[:length,:length])
        seq_pair_i = self.seq_pair_i(seq)
        seq_pair_j = self.seq_pair_j(seq)
        pair_embed = pair_embed + pos_embed + seq_pair_i[:,None,:,:] + seq_pair_j[:,:,None,:] # [batch_size,seq_len,seq_len,pair_embed_dim]

        return seq_embed,pair_embed
    
class SeqOuterProductMean(nn.Module):
    '''
    Parameters:
        seq_dim: int, the dimension of the input sequence
        pair_dim: int, the dimension of the output pair
        hidden_dim: int, the dimension of the hidden layer

    Inputs:
        seq: [batch_size,seq_len,seq_dim]

    Outputs:
        pair: [batch_size,seq_len,seq_len,pair_dim]
    '''
    def __init__(self,seq_dim,pair_dim,hidden_dim):
        super(SeqOuterProductMean,self).__init__()
        self.seq_dim = seq_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(self.seq_dim)
        self.l1 = nn.Linear(self.seq_dim,self.hidden_dim)
        self.l2 = nn.Linear(self.seq_dim,self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim*self.hidden_dim,self.pair_dim)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)
        nn.init.xavier_normal_(self.l3.weight)

    def forward(self,seq):
        batch,length,embed_dim = seq.shape
        seq = self.ln(seq)
        seq_1 = self.l1(seq)
        seq_2 = self.l2(seq)

        seq = torch.einsum('bid,bjf->bijdf',seq_1,seq_2)
        seq = seq.view(batch,length,length,-1)
        pair = self.l3(seq)

        return pair

class SeqAttwithPairBias(nn.Module):
    '''
    Parameters:
        seq_dim: int, the dimension of the input sequence
        pair_dim: int, the dimension of the input pair
        q_dim: int, the hidden dimension of the query
        k_dim: int, the hidden dimension of the key
        v_dim: int, the hidden dimension of the value
        num_heads: int, the number of the heads
        device: torch.device, device to run the model
    Inputs:
        seq: [batch_size,seq_len,seq_dim]
        pair: [batch_size,seq_len,seq_len,pair_dim]
    Outputs:
        seq: [batch_size,seq_len,seq_dim]
        current_kv: [batch_size,num_heads,seq_len,k_dim,v_dim]
    '''
    def __init__(self,seq_dim,pair_dim,q_dim,k_dim,v_dim,num_heads,device):
        super(SeqAttwithPairBias,self).__init__()
        self.seq_dim = seq_dim
        self.pair_dim = pair_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.device = device

        self.W_q = nn.Linear(self.seq_dim,self.q_dim*self.num_heads)
        self.W_k = nn.Linear(self.seq_dim,self.k_dim*self.num_heads)
        self.W_v = nn.Linear(self.seq_dim,self.v_dim*self.num_heads)

        self.ln_seq = nn.LayerNorm(self.seq_dim)
        self.ln_pair = nn.LayerNorm(self.pair_dim)
        self.W_b_i = nn.Linear(self.pair_dim,self.k_dim*self.num_heads)
        self.W_b_j = nn.Linear(self.pair_dim,self.v_dim*self.num_heads)

        self.W_g = nn.Linear(self.seq_dim,self.v_dim*self.num_heads)
        self.W_o = nn.Linear(self.v_dim*self.num_heads,self.seq_dim)

        self.decay_mask = torch.zeros(self.num_heads,1,1).to(self.device)
        self.biuld_decay_mask()
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.W_b_i.weight)
        nn.init.xavier_normal_(self.W_b_j.weight)
        nn.init.xavier_normal_(self.W_g.weight)
        nn.init.xavier_normal_(self.W_o.weight)

    def biuld_decay_mask(self):
        for h in range(self.num_heads):
            self.decay_mask[h][0][0] = 1 - 2 ** (-5-h)
    
    def forward(self,seq,pair,past_kv=0):
        b,length,_ = seq.shape
        
        # pair: [batch_size,seq_len,seq_len,pair_embed_dim]
        pair_i = pair.mean(dim=1) # [batch_size,seq_len,pair_embed_dim]
        pair_j = pair.mean(dim=2) # [batch_size,seq_len,pair_embed_dim]

        pair_i = self.W_b_i(pair_i) # [batch_size,seq_len,num_heads*k_dim]
        pair_j = self.W_b_j(pair_j) # [batch_size,seq_len,num_heads*v_dim]

        pair_i = pair_i.view(b,length,self.num_heads,self.k_dim) # [batch_size,seq_len,num_heads,k_dim]
        pair_j = pair_j.view(b,length,self.num_heads,self.v_dim) # [batch_size,seq_len,num_heads,v_dim]

        pair_i = pair_i.permute(0,2,1,3) # [batch_size,num_heads,seq_len,k_dim]
        pair_j = pair_j.permute(0,2,1,3) # [batch_size,num_heads,seq_len,v_dim]

        kv_bias = torch.einsum('bnlk,bnlv->bnlkv',pair_i,pair_j) # [batch_size,num_heads,seq_len,k_dim,v_dim]
        past_kv = past_kv + kv_bias

        seq = self.ln_seq(seq)
        pair = self.ln_pair(pair)

        gate = torch.sigmoid(self.W_g(seq)) # [batch_size,seq_len,num_heads*v_dim]
        gate = gate.contiguous().view(b,length,self.num_heads,self.v_dim) # [batch_size,seq_len,num_heads,v_dim]
        gate = gate.permute(0,2,1,3) # [batch_size,num_heads,seq_len,v_dim]

        Q = self.W_q(seq) # [batch_size,seq_len,num_heads*q_dim]
        K = self.W_k(seq) # [batch_size,seq_len,num_heads*k_dim]
        V = self.W_v(seq) # [batch_size,seq_len,num_heads*v_dim]

        Q = Q.contiguous().view(b,length,self.num_heads,self.q_dim) # [batch_size,seq_len,num_heads,q_dim]
        K = K.contiguous().view(b,length,self.num_heads,self.k_dim) # [batch_size,seq_len,num_heads,k_dim]
        V = V.contiguous().view(b,length,self.num_heads,self.v_dim) # [batch_size,seq_len,num_heads,v_dim]

        Q = Q.permute(0,2,1,3) # [batch_size,num_heads,seq_len,q_dim]
        K = K.permute(0,2,1,3) # [batch_size,num_heads,seq_len,k_dim]
        V = V.permute(0,2,1,3) # [batch_size,num_heads,seq_len,v_dim]

        Q = Q.unsqueeze(-1) # [batch_size,num_heads,seq_len,q_dim,1]

        current_kv = torch.einsum('bnlk,bnlv->bnlkv',K,V) # [batch_size,num_heads,seq_len,k_dim,v_dim]
        decay_mask = self.decay_mask.unsqueeze(0).unsqueeze(-1) # [1,num_heads,1,1,1]
        current_kv = decay_mask * past_kv + current_kv # [batch_size,num_heads,seq_len,k_dim,v_dim]

        seq = torch.sum(Q*current_kv,dim=-2) # [batch_size,num_heads,seq_len,v_dim]
        seq = F.group_norm(seq,seq.shape[1]) # [batch_size,num_heads,seq_len,v_dim]
        seq = gate*torch.sigmoid(gate)*seq # [batch_size,num_heads,seq_len,v_dim]
        seq = seq.permute(0,2,1,3) # [batch_size,seq_len,num_heads,v_dim]
        seq = seq.contiguous().view(b,length,-1) # [batch_size,seq_len,num_heads*v_dim]
        seq = self.W_o(seq) # [batch_size,seq_len,seq_dim]

        return seq,current_kv

class Triangle_Att_Start(nn.Module):
    def __init__(self,pair_dim,num_heads,hidden_dim):
        super(Triangle_Att_Start,self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(pair_dim)
        self.W_q = nn.Linear(pair_dim,hidden_dim*num_heads,bias=False)
        self.W_k = nn.Linear(pair_dim,hidden_dim*num_heads,bias=False)
        self.W_v = nn.Linear(pair_dim,hidden_dim*num_heads,bias=False)
        self.W_b = nn.Linear(pair_dim,num_heads,bias=False)
        self.W_g = nn.Linear(pair_dim,hidden_dim*num_heads)
        self.W_o = nn.Linear(hidden_dim*num_heads,pair_dim)
        self.decay_mask = torch.zeros(self.num_heads,1,1)

        self.factor = 1/math.sqrt(hidden_dim)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.W_q.weight)
        nn.init.xavier_normal_(self.W_k.weight)
        nn.init.xavier_normal_(self.W_v.weight)
        nn.init.xavier_normal_(self.W_b.weight)
        nn.init.xavier_normal_(self.W_g.weight)
        nn.init.xavier_normal_(self.W_o.weight)

    def biuld_decay_mask(self):
        for h in range(self.num_heads):
            self.decay_mask[h][0][0] = 1 - 2 ** (-5-h)

    def forward(self,pair,past_kv=0):
        batch,length, _, pair_dim = pair.shape
        
        pair = self.ln(pair)
        
        Q = self.W_q(pair) # [batch_size,seq_len,seq_len,num_heads*hidden_dim]
        K = self.W_k(pair) # [batch_size,seq_len,seq_len,num_heads*hidden_dim]
        V = self.W_v(pair) # [batch_size,seq_len,seq_len,num_heads*hidden_dim]

        Q = Q.contiguous().view(batch,length,length,self.num_heads,self.hidden_dim) # [batch_size,seq_len,seq_len,num_heads,hidden_dim]
        K = K.contiguous().view(batch,length,length,self.num_heads,self.hidden_dim) # [batch_size,seq_len,seq_len,num_heads,hidden_dim]
        V = V.contiguous().view(batch,length,length,self.num_heads,self.hidden_dim) # [batch_size,seq_len,seq_len,num_heads,hidden_dim]

        # retention
        current_kv = torch.einsum('bijnk,bijnv->bnijkv',K,V)  # [batch_size,num_heads,seq_len,seq_len,hidden_dim,hidden_dim]
        bias = self.W_b(pair) # [batch_size,seq_len,seq_len,num_heads]
        bias = bias.permute(0,3,2,1).unsqueeze(-1) # [batch_size,num_heads,seq_len,seq_len,1]
        bias = torch.einsum('bnijo,bijnv->bnijv',bias,V) # [batch_size,num_heads,seq_len,seq_len,hidden_dim]

        decay_mask = self.decay_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1,num_heads,1,1,1,1]
        current_kv = past_kv*decay_mask + current_kv # [batch_size,num_heads,seq_len,seq_len,hidden_dim,hidden_dim]
        
        Q = Q.permute(0,3,1,2,4) # [batch_size,num_heads,seq_len,seq_len,hidden_dim]
        output = torch.sum(Q.unsqueeze(-1) * current_kv,dim=-2) # [batch_size,num_heads,seq_len,hidden_dim]
        output = output + bias # [batch_size,num_heads,seq_len,seq_len,hidden_dim]
        output = F.group_norm(output,output.shape[1]) # [batch_size,num_heads,seq_len,seq_len,hidden_dim]

        gate = torch.sigmoid(self.W_g(pair))
        gate = gate.contiguous().view(batch,length,length,self.num_heads,self.hidden_dim) # [batch_size,seq_len,seq_len,num_heads,hidden_dim]
        gate = gate.permute(0,3,1,2,4) # [batch_size,num_heads,seq_len,seq_len,hidden_dim]
        output = gate*torch.sigmoid(gate)*output
        output = output.permute(0,2,3,1,4).contiguous().view(batch,length,length,-1) # [batch_size,seq_len,seq_len,num_heads*hidden_dim]
        output = self.W_o(output)
        return pair,current_kv

class Triangle_Att_End(nn.Module):
    def __init__(self,pair_dim,num_heads,hidden_dim):
        super(Triangle_Att_End,self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(pair_dim)
        self.W_q = nn.Linear(pair_dim,hidden_dim*num_heads,bias=False)
        self.W_k = nn.Linear(pair_dim,hidden_dim*num_heads,bias=False)
        self.W_v = nn.Linear(pair_dim,hidden_dim*num_heads,bias=False)
        self.W_b = nn.Linear(pair_dim,num_heads,bias=False)
        self.W_g = nn.Linear(pair_dim,hidden_dim*num_heads)
        self.W_o = nn.Linear(hidden_dim*num_heads,pair_dim)
        self.decay_mask = torch.zeros(self.num_heads,1,1)

        self.factor = 1/math.sqrt(hidden_dim)
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.W_q.weight)
        nn.init.xavier_normal_(self.W_k.weight)
        nn.init.xavier_normal_(self.W_v.weight)
        nn.init.xavier_normal_(self.W_b.weight)
        nn.init.xavier_normal_(self.W_g.weight)
        nn.init.xavier_normal_(self.W_o.weight)

    def biuld_decay_mask(self):
        for h in range(self.num_heads):
            self.decay_mask[h][0][0] = 1 - 2 ** (-5-h)

    def forward(self,pair,past_kv=0):
        batch,length, _, pair_dim = pair.shape
        
        pair = self.ln(pair)
        
        Q = self.W_q(pair) # [batch_size,seq_len,seq_len,num_heads*hidden_dim]
        K = self.W_k(pair) # [batch_size,seq_len,seq_len,num_heads*hidden_dim]
        V = self.W_v(pair) # [batch_size,seq_len,seq_len,num_heads*hidden_dim]

        Q = Q.contiguous().view(batch,length,length,self.num_heads,self.hidden_dim) # [batch_size,seq_len,seq_len,num_heads,hidden_dim]
        K = K.contiguous().view(batch,length,length,self.num_heads,self.hidden_dim) # [batch_size,seq_len,seq_len,num_heads,hidden_dim]
        V = V.contiguous().view(batch,length,length,self.num_heads,self.hidden_dim) # [batch_size,seq_len,seq_len,num_heads,hidden_dim]

        # retention
        current_kv = torch.einsum('bijnk,bijnv->bnijkv',K,V)  # [batch_size,num_heads,seq_len,seq_len,hidden_dim,hidden_dim]
        bias = self.W_b(pair) # [batch_size,seq_len,seq_len,num_heads]
        bias = bias.permute(0,3,2,1).unsqueeze(-1) # [batch_size,num_heads,seq_len,seq_len,1]
        bias = torch.einsum('bnijo,bijnv->bnijo',bias,V) # [batch_size,num_heads,seq_len,seq_len,1]

        decay_mask = self.decay_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1,num_heads,1,1,1,1]
        current_kv = decay_mask * past_kv + current_kv # [batch_size,num_heads,seq_len,seq_len,hidden_dim,hidden_dim]

        Q = Q.permute(0,3,1,2,4) # [batch_size,num_heads,seq_len,seq_len,hidden_dim]
        output = torch.sum(Q.unsqueeze(-1) * current_kv,dim=-2) # [batch_size,num_heads,seq_len,hidden_dim]
        output = output + bias # [batch_size,num_heads,seq_len,seq_len,hidden_dim]
        output = F.group_norm(output,output.shape[1]) # [batch_size,num_heads,seq_len,seq_len,hidden_dim]

        gate = torch.sigmoid(self.W_g(pair))
        gate = gate.contiguous().view(batch,length,length,self.num_heads,self.hidden_dim) # [batch_size,seq_len,seq_len,num_heads,hidden_dim]
        gate = gate.permute(0,3,1,2,4) # [batch_size,num_heads,seq_len,seq_len,hidden_dim]
        output = gate*torch.sigmoid(gate)*output
        output = output.permute(0,2,3,1,4).contiguous().view(batch,length,length,-1) # [batch_size,seq_len,seq_len,num_heads*hidden_dim]
        output = self.W_o(output)
        
        return pair,current_kv

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

        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.W_a.weight)
        nn.init.xavier_normal_(self.gate_a.weight)
        nn.init.xavier_normal_(self.W_b.weight)
        nn.init.xavier_normal_(self.gate_b.weight)
        nn.init.xavier_normal_(self.W_o.weight)
        nn.init.xavier_normal_(self.W_g.weight)
    
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

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.W_a.weight)
        nn.init.xavier_normal_(self.gate_a.weight)
        nn.init.xavier_normal_(self.W_b.weight)
        nn.init.xavier_normal_(self.gate_b.weight)
        nn.init.xavier_normal_(self.W_o.weight)
        nn.init.xavier_normal_(self.W_g.weight)

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
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)
        
    def forward(self,pair):
        pair = self.ln(pair)
        pair = self.l1(pair)
        pair = self.relu(pair)
        pair = self.l2(pair)
        
        return pair

class RetentionEncoder(nn.Module):

    def __init__(self,seq_dim,pair_dim,hidden_dim,num_heads,device):
        super(RetentionEncoder,self).__init__()

        self.seq_dim = seq_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.triangle_att_start = Triangle_Att_Start(self.pair_dim,self.num_heads,self.hidden_dim)
        self.triangle_att_end = Triangle_Att_End(self.pair_dim,self.num_heads,self.hidden_dim)
        self.triangle_update_outgoing = Triangle_Update_Outgoing(self.pair_dim,self.hidden_dim)
        self.triangle_update_incoming = Triangle_Update_Incoming(self.pair_dim,self.hidden_dim)
        self.pair_transition = Pair_Transition(self.pair_dim,self.hidden_dim)

        self.seq_att_with_pair_bias = SeqAttwithPairBias(self.seq_dim,self.pair_dim,self.hidden_dim,self.hidden_dim,self.hidden_dim,self.num_heads,device)
        self.seq_outer_product_mean = SeqOuterProductMean(self.seq_dim,self.pair_dim,self.hidden_dim)

    def forward(self,seq,pair,past_kv_seq=0,past_kv_pair=0):
        pair,current_kv_pair = self.triangle_att_start(pair,past_kv_pair)
        pair = self.triangle_update_outgoing(pair)
        pair,current_kv_pair = self.triangle_att_end(pair,current_kv_pair)
        pair = self.triangle_update_incoming(pair)
        pair = self.pair_transition(pair)
        seq,current_kv_seq = self.seq_att_with_pair_bias(seq,pair,past_kv_seq)
        pair_bias = self.seq_outer_product_mean(seq)
        pair = pair + pair_bias
        return seq,pair,current_kv_seq,current_kv_pair

class RetentionFold(nn.Module):

    def __init__(self,seq_dim,pair_dim,hidden_dim,num_heads,n_layers,device):
        super(RetentionFold,self).__init__()

        self.seq_dim = seq_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.input_embedding = SeqEmbed(self.seq_dim,self.pair_dim,self.hidden_dim,device,14,65)
        self.retention_encoder = nn.ModuleList()

        for i in range(n_layers):
            self.retention_encoder.append(RetentionEncoder(self.hidden_dim,self.pair_dim,self.hidden_dim,self.num_heads,device))
    
    def forward(self,seq):
        seq,pair = self.input_embedding(seq)
        past_kv_seq = 0
        past_kv_pair = 0
        for i in range(len(self.retention_encoder)):
            seq,pair,past_kv_seq,past_kv_pair = self.retention_encoder[i](seq,pair,past_kv_seq,past_kv_pair)
        data = tuple([seq,pair])
        return data
