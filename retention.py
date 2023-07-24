import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelRetention(nn.Module):

    def __init__(self,embed_dim,q_dim,k_dim,v_dim,num_heads,max_length=128):
        super(ParallelRetention,self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.W_Q = nn.Linear(self.embed_dim,self.q_dim)
        self.W_K = nn.Linear(self.embed_dim,self.k_dim)
        self.W_V = nn.Linear(self.embed_dim,self.v_dim)

        self.max_length = max_length
        self.decay_mask = self.biuld_decay_mask()
        self.init_parameters()

    def biuld_decay_mask(self):
        decay_mask = torch.zeros(self.num_heads,self.max_length,self.max_length)
        for h in range(self.num_heads):
            decay = 1 - 2 ** (-5-h) 
            for i in range(self.max_length):
                for j in range(i,self.max_length):
                    decay_mask[h][j][i] = decay ** (i-j)
        return decay_mask
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_V.weight)


    def forward(self,x):
        '''
        x: [batch_size,num_heads,seq_len,embed_dim]
        Q: [batch_size,num_heads,seq_len,q_dim]
        K: [batch_size,num_heads,seq_len,k_dim]
        V: [batch_size,num_heads,seq_len,v_dim]
        decay_mask: [num_heads,seq_len,seq_len]
        '''
        b,n,l,e = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        D = self.decay_mask[:,:l,:l]
        output = parallel_retention(Q,K,V,D)

        return output

class RecurrentRetention(nn.Module):

    def __init__(self,embed_dim,q_dim,k_dim,v_dim,num_heads):
        
        super(RecurrentRetention,self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.W_Q = nn.Linear(self.embed_dim,self.q_dim)
        self.W_K = nn.Linear(self.embed_dim,self.k_dim)
        self.W_V = nn.Linear(self.embed_dim,self.v_dim)

        self.decay_mask = self.biuld_decay_mask()

    def init_parameters(self):
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_V.weight)

    def biuld_decay_mask(self):
        decay_mask = torch.zeros(self.num_heads,1,1)
        for h in range(self.num_heads):
            decay_mask[h][0][0] = 1 - 2 ** (-5-h)
        return decay_mask

    def forward(self,x,past_kv=0):
        b,n,l,e = x.shape
        Q = self.W_Q(x) # [batch_size,num_heads,seq_len,q_dim]
        K = self.W_K(x) # [batch_size,num_heads,seq_len,k_dim]
        V = self.W_V(x) # [batch_size,num_heads,seq_len,v_dim]
        output,current_kv = recurrent_retention(Q,K,V,past_kv,self.decay_mask) # [batch_size,seq_len,num_heads*v_dim]
        return output,current_kv

def recurrent_retention(q,k,v,
                       past_kv,
                       decay):
    '''
    q: [batch_size,num_heads,seq_len,q_dim]
    k: [batch_size,num_heads,seq_len,k_dim]
    v: [batch_size,num_heads,seq_len,v_dim]
    past_kv: [batch_size,num_heads,k_dim,v_dim]
    decay: [num_heads,1,1]
    '''
    b,n,l,_ = q.shape

    Q = q.unsqueeze(-2) # [batch_size,num_heads,seq_len,1,q_dim]
    K = k.unsqueeze(-1) # [batch_size,num_heads,seq_len,k_dim,1]
    V = v.unsqueeze(-2) # [batch_size,num_heads,seq_len,1,v_dim]

    current_kv = decay * past_kv + torch.einsum('bnlki,bnlvj->bnlkv') # [batch_size,num_heads,seq_len,k_dim,v_dim]
    output = torch.sum(Q * current_kv,dim=-2) # [batch_size,num_heads,seq_len,v_dim]
    output = F.group_norm(output,output.shape[1]) # [batch_size,num_heads,seq_len,v_dim]
    
    return output,current_kv

def parallel_retention(q,k,v,
                       decay):
    '''
    q: [batch_size,num_heads,seq_len,q_dim]
    k: [batch_size,num_heads,seq_len,k_dim]
    v: [batch_size,num_heads,seq_len,v_dim]
    decay: [num_heads,seq_len,seq_len]
    '''
    b,num_heads,l,_ = q.shape

    Q = q.unsqueeze(-1) # [batch_size,num_heads,seq_len,q_dim,1]
    K = k.unsqueeze(-2) # [batch_size,num_heads,seq_len,1,k_dim]
    V = v.unsqueeze(-1) # [batch_size,num_heads,seq_len,v_dim,1]

    output = torch.matmul(Q,K) * decay # [batch_size,seq_len,q_dim,k_dim]
    output = torch.matmul(output,V) # [batch_size,seq_len,q_dim,v_dim]
    output = F.group_norm(output,output.shape[1]) # [batch_size,seq_len,q_dim,v_dim]
    output = output.reshape(b,l,-1) # [batch_size,seq_len,num_heads*v_dim]

    return output
