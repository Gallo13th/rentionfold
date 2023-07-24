import numpy as np

NT_EMBEDDING = {
    'A' : [1,0,0,0,0,0],
    'C' : [0,1,0,0,0,0],
    'G' : [0,0,1,0,0,0],
    'U' : [0,0,0,1,0,0],
    'I' : [0,0,0,0,1,0],
    'X' : [0,0,0,0,0,1],
    'N' : [pow(6,-0.5) for _ in range(6)],
    'V' : [pow(3,-0.5) for _ in range(3)] + [0,0,0]
}

def seq_one_hot_encode(seq):
    return [NT_EMBEDDING[nt] for nt in seq]

def data_generateor()