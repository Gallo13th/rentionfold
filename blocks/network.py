import torch
from torch import nn

from blocks import encoder,embed,pair,head
from torch.utils.checkpoint import checkpoint_sequential
class SingleSeqTransformer(nn.Module):
    
    def __init__(self,input_seq_dim,seq_dim,pair_dim,n_heads,hidden_dim,n_layers_1,n_layers_2,device):
        super(SingleSeqTransformer, self).__init__()
        # embed layer
        self.input_embed = embed.InputEmbed_SingleSeq(input_seq_dim,1,seq_dim,pair_dim,14,65,device)
        
        self.transformer_1 = torch.nn.Sequential(*[encoder.Encoder_SigleSeqWithPairBias(seq_dim,pair_dim,n_heads,hidden_dim) for _ in range(n_layers_1)])
        self.transformer_2 = torch.nn.Sequential(*[encoder.Encoder_SigleSeqWithoutPairBias(seq_dim,pair_dim,n_heads,hidden_dim) for _ in range(n_layers_2)])
        
        self.constraint_head = head.ConstraintHead(pair_dim)
        self.seq_predict = torch.nn.Sequential(
            nn.Linear(in_features=seq_dim,out_features=36),
            nn.GELU(),
            nn.Linear(in_features=36,out_features=9)
        )
        
    def forward(self,seq,pair):
        b,l,_ = seq.shape
        
        pair = pair.permute(0,2,3,1)
        seq,pair = self.input_embed(seq,pair)
        seq,pair = self.transformer_1([seq,pair])
        seq,pair = self.transformer_2([seq,pair])
        seq = self.seq_predict(seq)
        dist_C4,dist_C1,dist_N,Omega,Theta,Phi = self.constraint_head(pair)
        return [seq,dist_C4,dist_C1,dist_N,Omega,Theta,Phi]
    
class MSATransformer(nn.Module):
    
    def __init__(self,input_seq_dim,input_pair_dim,output_msa_dim_1,output_msa_dim_2,output_pair_dim,n_heads,hidden_dim,n_layers_1,n_layers_2,device):
        super(MSATransformer,self).__init__()
        self.n_seqs = output_msa_dim_1
        self.msa_dim = output_msa_dim_2
        
        # embed layer
        self.input_embed = embed.InputEmbed_MSAlike(input_seq_dim,input_pair_dim,output_msa_dim_1,output_msa_dim_2,output_pair_dim,14,65,device)
        
        self.transformer_1 = torch.nn.Sequential(*[encoder.Encoder_MSAWithPairBias(self.msa_dim,output_pair_dim,n_heads,hidden_dim) for _ in range(n_layers_1)])
        self.transformer_2 = torch.nn.Sequential(*[encoder.Encoder_SigleSeqWithoutPairBias(self.msa_dim,output_pair_dim,n_heads,hidden_dim) for _ in range(n_layers_2)])
        
        self.constraint_head = head.ConstraintHead(output_pair_dim)
        self.seq_predict = head.BackboneHead(self.msa_dim)
        
        
    def forward(self,seq):
        b,l,_ = seq.shape
        msa,pair = self.input_embed(seq)
        msa,pair = checkpoint_sequential(self.transformer_1,12,[msa,pair])
        seq = msa[:,0,:,:]
        seq,pair = self.transformer_2([seq,pair])
        seq_pred = self.seq_predict(seq)
        dist_C4,dist_P,dist_N,Omega,Lambda = self.constraint_head(pair)
        return [seq_pred,dist_C4,dist_P,dist_N,Omega,Lambda]
    
        