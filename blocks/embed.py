import torch
import numpy as np
from torch import nn
from blocks import pair

class SingleSeq_To_SingleSeq_Embed(nn.Module):
    
    def __init__(self,input_embed,output_embed):
        super(SingleSeq_To_SingleSeq_Embed,self).__init__()
        self.stem = torch.nn.Linear(in_features=input_embed,out_features=output_embed)
        self.ln = torch.nn.LayerNorm(output_embed)
        
    def forward(self,seq):
        return self.ln(self.stem(seq))
    
class SingleSeq_To_MSAlike_Embed(nn.Module):
    
    def __init__(self,input_embed,output_n_heads,output_hidden_dim):
        super(SingleSeq_To_MSAlike_Embed,self).__init__()
        self.n_heads = output_n_heads
        self.hidden_dim = output_hidden_dim
        self.stem = torch.nn.Linear(in_features=input_embed,out_features=output_hidden_dim*output_n_heads)
        self.ln = torch.nn.LayerNorm(output_hidden_dim*output_n_heads)
        
    def forward(self,seq):
        batch,length,_ = seq.shape
        seq = self.ln(self.stem(seq))
        seq = seq.view(batch,length,self.n_heads,self.hidden_dim)
        return seq
    
class InputEmbed_SingleSeq(nn.Module):
    
    def __init__(self,input_seq_dim,input_pair_dim,output_seq_dim,output_pair_dim,n_pos_bins_seq,n_pos_bins_pair, device):
        super(InputEmbed_SingleSeq,self).__init__()
        self.seq_embed = SingleSeq_To_SingleSeq_Embed(input_seq_dim,output_seq_dim)
        self.stem = torch.nn.Linear(in_features=input_pair_dim,out_features=output_pair_dim)
        self.pos_1 = self.pos_embed_pair(device,n_pos_bins_pair)
        self.pos_2 = self.pos_embed_seq(device,n_pos_bins_seq)
        self.pos_pair_embed = torch.nn.Linear(in_features=n_pos_bins_pair,out_features=output_pair_dim)
        self.pos_seq_embed = torch.nn.Linear(in_features=n_pos_bins_seq,out_features=output_seq_dim)
        self.pair_transition = pair.Pair_Transition(output_pair_dim,2*output_pair_dim)
        self.seq_pair_i = torch.nn.Linear(in_features=input_seq_dim,out_features=output_pair_dim)
        self.seq_pair_j = torch.nn.Linear(in_features=input_seq_dim,out_features=output_pair_dim)
        
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
    
    def forward(self,seq,pair):
        batch,length,_ = seq.shape
        pair = self.stem(pair)
        pos_embed = self.pos_pair_embed(self.pos_1[:length,:length])
        seq_pair_i = self.seq_pair_i(seq)
        seq_pair_j = self.seq_pair_j(seq)
        pair = pair + pos_embed + seq_pair_i[:,None,:,:] + seq_pair_j[:,None,:,:]
        pair = self.pair_transition(pair)
        
        seq = self.seq_embed(seq)
        pos_embed = self.pos_seq_embed(self.pos_2[:length])
        seq = seq + pos_embed
        
        return seq,pair

class InputEmbed_MSAlike(nn.Module):
    
    def __init__(self,input_seq_dim,input_pair_dim,output_msa_dim_1,output_msa_dim_2,output_pair_dim,n_pos_bins_seq,n_pos_bins_pair, device):
        super(InputEmbed_MSAlike,self).__init__()
        self.n_seqs = output_msa_dim_1
        self.msa_dim = output_msa_dim_2
        self.seq_embed = SingleSeq_To_SingleSeq_Embed(input_seq_dim,output_msa_dim_1*output_msa_dim_2)
        self.stem = torch.nn.Linear(in_features=input_pair_dim,out_features=output_pair_dim)
        self.pos_1 = self.pos_embed_pair(n_pos_bins_pair,device)
        self.pos_2 = self.pos_embed_seq(n_pos_bins_seq,device)
        self.pos_pair_embed = torch.nn.Linear(in_features=n_pos_bins_pair,out_features=output_pair_dim)
        self.pos_seq_embed = torch.nn.Linear(in_features=n_pos_bins_seq,out_features=output_msa_dim_2)
        self.pair_transition = pair.Pair_Transition(output_pair_dim,2*output_pair_dim)
        self.seq_pair_i = torch.nn.Linear(in_features=input_seq_dim,out_features=output_pair_dim)
        self.seq_pair_j = torch.nn.Linear(in_features=input_seq_dim,out_features=output_pair_dim)
        self.seq_msa = torch.nn.Linear(in_features=input_seq_dim,out_features=output_msa_dim_2)
        
    def pos_embed_seq(self,n_pos_bins,device):
        pos = torch.arange(512)
        rel_pos = ((pos[:,None] & (1 << np.arange(n_pos_bins)))) > 0
        return rel_pos.float().to(device)
    
    def pos_embed_pair(self,n_pos_bins,device):
        pos = torch.arange(512)
        rel_pos = pos[None,:]-pos[:,None]
        rel_pos = rel_pos.clamp(-32,32)
        rel_pos_encode = torch.nn.functional.one_hot(rel_pos+32,n_pos_bins)
        return rel_pos_encode.float().to(device)
    
    def forward(self,seq):
        batch,length,_ = seq.shape
        #pair = self.stem(pair)
        pos_embed = self.pos_pair_embed(self.pos_1[:length,:length])
        seq_pair_i = self.seq_pair_i(seq)
        seq_pair_j = self.seq_pair_j(seq)
        pair = pos_embed + seq_pair_i[:,None,:,:] + seq_pair_j[:,None,:,:]
        pair = self.pair_transition(pair)
        
        msa = self.seq_embed(seq)
        msa = msa.contiguous().view(batch,length,self.n_seqs,self.msa_dim)
        msa = msa.permute(0,2,1,3).contiguous()
        
        seq_msa = self.seq_msa(seq)
        pos_embed = self.pos_seq_embed(self.pos_2[:length])
        msa = msa + pos_embed[None,None,:,:] + seq_msa[:,None,:,:]
        
        return msa,pair