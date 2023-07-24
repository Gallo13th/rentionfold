import torch
from torch import nn

from blocks import seq,pair,head,embed,msa


class PairEncoder(nn.Module):
    def __init__(self,pair_dim,n_heads,hidden_dim):
        super(PairEncoder,self).__init__()
        
        self.triangle_update_outgoing = pair.Triangle_Update_Outgoing(pair_dim,hidden_dim)
        self.triangle_update_incoming = pair.Triangle_Update_Incoming(pair_dim,hidden_dim)
        self.triangle_att_start = pair.Triangle_Att_Start(pair_dim,n_heads,hidden_dim)
        self.triangle_att_end  = pair.Triangle_Att_End(pair_dim,n_heads,hidden_dim)
        self.pair_transition = pair.Pair_Transition(pair_dim,4*pair_dim)
        
    def forward(self,pair):
        pair = pair + self.triangle_update_outgoing(pair)
        pair = pair + self.triangle_update_incoming(pair)
        pair = pair + self.triangle_att_start(pair)
        pair = pair + self.triangle_att_end(pair)
        pair = pair + self.pair_transition(pair)
        return pair

class SeqEncoder(nn.Module):
    def __init__(self,seq_dim,pair_dim,n_heads,hidden_dim):
        super(SeqEncoder, self).__init__()

        self.seqatt_with_pairbias = seq.SeqAttwithPairBias(seq_dim,pair_dim,n_heads,hidden_dim)
        self.att_seq = seq.SeqAtt(seq_dim,n_heads,hidden_dim)
        self.seq_trans = seq.SeqTransition(seq_dim)

    def forward(self,seq,pair):
        seq = seq + self.seqatt_with_pairbias(seq,pair)
        seq = seq + self.att_seq(seq)
        seq = seq + self.seq_trans(seq)

        return seq

class Encoder_SigleSeqWithPairBias(nn.Module):
    def __init__(self,seq_dim,pair_dim,n_heads,hidden_dim):
        super(Encoder_SigleSeqWithPairBias, self).__init__()
        # pair
        self.triangle_update_outgoing = pair.Triangle_Update_Outgoing(pair_dim,hidden_dim)
        self.triangle_update_incoming = pair.Triangle_Update_Incoming(pair_dim,hidden_dim)
        self.triangle_att_start = pair.Triangle_Att_Start(pair_dim,n_heads,hidden_dim)
        self.triangle_att_end  = pair.Triangle_Att_End(pair_dim,n_heads,hidden_dim)
        self.pair_transition = pair.Pair_Transition(pair_dim,4*pair_dim)
        # seq
        self.seqatt_with_pairbias = seq.SeqAttwithPairBias(seq_dim,pair_dim,n_heads,hidden_dim)
        self.att_seq = seq.SeqAtt(seq_dim,n_heads,hidden_dim)
        self.seq_trans = seq.SeqTransition(seq_dim)
        # out_product
        self.ln = nn.LayerNorm(seq_dim)
        self.l1 = nn.Linear(seq_dim,hidden_dim)
        self.l2 = nn.Linear(seq_dim,hidden_dim)
        self.l = nn.Linear(hidden_dim*hidden_dim,pair_dim)
        
    def forward(self,x):
        seq,pair = x
        batch,length,_ = seq.shape
        seq = seq + self.seqatt_with_pairbias(seq,pair)
        seq = seq + self.att_seq(seq)
        seq = seq + self.seq_trans(seq)
        
        pair_bias = self.ln(seq)
        pair_bias = torch.einsum('bim,bjn->bijmn',self.l1(pair_bias),self.l2(pair_bias))
        pair_bias = torch.mean(pair_bias,dim=0)
        pair_bias = pair_bias.view(1,length,length,-1)
        pair_bias = self.l(pair_bias)
        pair = pair + pair_bias
        pair = pair + self.triangle_update_outgoing(pair)
        pair = pair + self.triangle_update_incoming(pair)
        pair = pair + self.triangle_att_start(pair)
        pair = pair + self.triangle_att_end(pair)
        pair = pair + self.pair_transition(pair)

        return [seq,pair]
    
class Encoder_SigleSeqWithoutPairBias(nn.Module):
    def __init__(self,seq_dim,pair_dim,n_heads,hidden_dim):
        super(Encoder_SigleSeqWithoutPairBias, self).__init__()
        # pair
        self.triangle_update_outgoing = pair.Triangle_Update_Outgoing(pair_dim,hidden_dim)
        self.triangle_update_incoming = pair.Triangle_Update_Incoming(pair_dim,hidden_dim)
        self.triangle_att_start = pair.Triangle_Att_Start(pair_dim,n_heads,hidden_dim)
        self.triangle_att_end  = pair.Triangle_Att_End(pair_dim,n_heads,hidden_dim)
        self.pair_transition = pair.Pair_Transition(pair_dim,4*pair_dim)
        # seq
        self.seqatt_with_pairbias = seq.SeqAttwithPairBias(seq_dim,pair_dim,n_heads,hidden_dim)
        self.att_seq = seq.SeqAtt(seq_dim,n_heads,hidden_dim)
        self.seq_trans = seq.SeqTransition(seq_dim)

        
    def forward(self, x):
        seq,pair = x
        batch,length,_ = seq.shape
        seq = seq + self.seqatt_with_pairbias(seq,pair)
        seq = seq + self.att_seq(seq)
        seq = seq + self.seq_trans(seq)
        
        pair = pair + self.triangle_update_outgoing(pair)
        pair = pair + self.triangle_update_incoming(pair)
        pair = pair + self.triangle_att_start(pair)
        pair = pair + self.triangle_att_end(pair)
        pair = pair + self.pair_transition(pair)

        return [seq,pair]

class Encoder_MSAWithPairBias(nn.Module):
    
    def __init__(self,msa_dim,pair_dim,n_heads,hidden_dim):
        super(Encoder_MSAWithPairBias,self).__init__()
        # pair
        self.triangle_update_outgoing = pair.Triangle_Update_Outgoing(pair_dim,hidden_dim)
        self.triangle_update_incoming = pair.Triangle_Update_Incoming(pair_dim,hidden_dim)
        self.triangle_att_start = pair.Triangle_Att_Start(pair_dim,n_heads,hidden_dim)
        self.triangle_att_end  = pair.Triangle_Att_End(pair_dim,n_heads,hidden_dim)
        self.pair_transition = pair.Pair_Transition(pair_dim,4*pair_dim)
        
        # msa
        self.msa_row_att = msa.MSA_Row_Att(msa_dim,pair_dim,n_heads,hidden_dim)
        self.msa_col_att = msa.MSA_Col_Att(msa_dim,n_heads,hidden_dim)
        self.msa_transition = msa.MSA_Transition(msa_dim,hidden_dim)
        self.msa_outer_product_mean = msa.MSA_Outer_Product_Mean(msa_dim,pair_dim,hidden_dim)
        
    def forward(self,x):
        msa,pair = x
        msa = msa + self.msa_row_att(msa,pair)
        msa = msa + self.msa_col_att(msa)
        msa = msa + self.msa_transition(msa)
        pair = pair + self.msa_outer_product_mean(msa)
        pair = pair + self.triangle_update_outgoing(pair)
        pair = pair + self.triangle_update_incoming(pair)
        pair = pair + self.triangle_att_start(pair)
        pair = pair + self.triangle_att_end(pair)
        pair = pair + self.pair_transition(pair)
        return [msa, pair]
