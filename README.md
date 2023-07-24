# Rentionfold
Try to use RetNet for Protein/NA (maybe with ligand) structure prediction

## Variant of Some Modules

### Recurrent TriAngleAttention

考虑三角自注意力的表达形式：
$$
output = softmax(attention\_ map + bias) @ value \\
attention\_ map = query @ key^T \\
$$
考虑recurrent retention的表达形式：
$$
output_n = query_n @ S_n \\
S_n = decay\_ mask S_{n-1} + key_n^T @ value_n
$$
对三角自注意力模块进行符合retention思路的变形：
$$
\begin{aligned}
output &= (query @ key^T + bias) @ value \\
&= query@key^T@value + bias@value \\
&= Retention(input) + bias@value \\
&= Retention(input) + Bias \\
\end{aligned}
$$
对于Start/End模块，需要分别处理Bias的具体形式。

### Start/End Blocks
在三角注意力的起始模块中，具体的差异来源于bias的拓展维度
$$
\begin{aligned}
attention\_ map_{...ij} &= (query @ key^T)_{...ij} + bias_{...i} \\
&or=(query @ key^T)_{...ij} + bias_{...j} \\
\end{aligned}
$$
依旧跳过$QK^T$的讨论，Bias的分量形式为：
$$
Bias = bias@value \\
\begin{aligned}
Bias_{...ij} &= \sum_{k} bias_{...i} \times value_{...kj} \\
or Bias_{....j}&=\sum_{k} bias_{...j} \times value_{...kj} \\
\end{aligned}
$$

于是，Start/End模块的伪算法如下：
```python
input = layer_norm(input)

Q = W_q(input) # (batch,length,length,n_heads*hidden_dim)
K = W_k(input) # (batch,length,length,n_heads*hidden_dim)
V = W_v(input) # (batch,length,length,n_heads*hidden_dim)

Q = Q.contiguous().view(batch,length,length,n_heads,hidden_dim) # (batch,length,length,n_heads,hidden_dim)
K = K.contiguous().view(batch,length,length,n_heads,hidden_dim) # (batch,length,length,n_heads,hidden_dim)
V = V.contiguous().view(batch,length,length,n_heads,hidden_dim) # (batch,length,length,n_heads,hidden_dim)

# retention
current_kv = torch.einsum('bijnk,bijnv->bijnkv',K,V)  # [batch_size,num_heads,seq_len,seq_len,hidden_dim,hidden_dim]
bias = W_b(input) # [batch_size,seq_len,seq_len,num_heads]
bias = bias.unsqueeze(-1) # [batch_size,seq_len,seq_len,num_heads,1]

# Start:
bias = torch.einsum('bnijo,bijnv->bnijv',bias,V) # [batch_size,num_heads,seq_len,seq_len,hidden_dim]
## End:
# bias = torch.einsum('bnijo,bijnv->bnijo',bias,V) # [batch_size,num_heads,seq_len,seq_len,1]

decay_mask = decay_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1,num_heads,1,1,1,1]
current_kv = decay_mask * past_kv + current_kv # [batch_size,num_heads,seq_len,seq_len,hidden_dim,hidden_dim]

output = Q@current_kv + bias # [batch_size,num_heads,seq_len,seq_len,hidden_dim]

gate = torch.sigmoid(W_g(input))
gate = gate.contiguous().view(batch,length,length,n_heads,hidden_dim) # [batch_size,seq_len,seq_len,num_heads,hidden_dim]

output = gate*torch.sigmoid(gate)*output
output = W_o(output)

```

