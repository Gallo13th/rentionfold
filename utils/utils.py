import torch
import random

NT_EMBEDDING = {
    'A': [1, 0, 0, 0, 0, 0],
    'C': [0, 1, 0, 0, 0, 0],
    'G': [0, 0, 1, 0, 0, 0],
    'U': [0, 0, 0, 1, 0, 0],
    'T': [0, 0, 0, 1, 0, 0],
    'I': [0, 0, 0, 0, 1, 0],
    'X': [0, 0, 0, 0, 0, 1],
    'N': [pow(6, -0.5) for _ in range(6)],
    'V': [pow(3, -0.5) for _ in range(3)] + [0, 0, 0],
    'Y': [0, pow(2, -0.5), 0, pow(2, -0.5), 0, 0],
    'K': [0, 0, pow(2, -0.5), pow(2, -0.5), 0, 0],
    'W': [pow(2, -0.5), 0, 0, pow(2, -0.5), 0, 0],
    'D': [pow(2, -0.5), 0, pow(2, -0.5),
          pow(2, -0.5), 0, 0],
    'R': [pow(2, -0.5), 0, pow(2, -0.5), 0, 0, 0],
    'S': [0, pow(2, -0.5), pow(2, -0.5), 0, 0, 0],
    'M': [pow(2, -0.5), pow(2, -0.5), 0, 0, 0, 0],
    '_': [0, 0, 0, 0, 0, 0]
}
NT_ONE_HOT = {
    'A': [1, 0, 0, 0, 0, 0],
    'C': [0, 1, 0, 0, 0, 0],
    'G': [0, 0, 1, 0, 0, 0],
    'U': [0, 0, 0, 1, 0, 0],
    'T': [0, 0, 0, 1, 0, 0],
    'I': [0, 0, 0, 0, 1, 0],
    'X': [0, 0, 0, 0, 0, 1],
    '_': [0, 0, 0, 0, 0, 0]
}
NT_Degeneracy = {
    'N': ['A', 'C', 'G', 'U'],
    'V': ['G', 'C', 'A'],
    'Y': ['U', 'C'],
    'K': ['G', 'U'],
    'W': ['A', 'U'],
    'D': ['G', 'A', 'U'],
    'R': ['G', 'A'],
    'S': ['G', 'C'],
    'M': ['A', 'C'],
    'A': ['A'],
    'C': ['C'],
    'G': ['G'],
    'U': ['U'],
    'T': ['T'],
    'X': ['X'],
    'I': ['I'],
    '_': ['_']
}

device = torch.device("cuda:0")

def seq_one_hot_encode(seq):
    return [NT_ONE_HOT[random.choice(NT_Degeneracy[nt.upper()])] for nt in seq]

def seq_embedding(seq):
    return [NT_EMBEDDING[nt.upper()] for nt in seq]
