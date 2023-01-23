import math
import torch
def create_positional_embedding(max_len, embed_dim):
    '''
    Args:
        max_len: The maximum length supported for positional embeddings
        embed_dim: The size of your embeddings
    Returns:
        pe: [max_len, 1, embed_dim] computed as in the formulae above
    '''
    pe = None

    ### TODO ###
    pe = torch.zeros([max_len, embed_dim], dtype=torch.float64)
    for k in range (max_len):
      for i in range(0,embed_dim,2):
        # denominator = math.exp(math.log(math.pow(10000,(2*i)/embed_dim)))
        pe[k, i] = math.sin(k/(math.exp(math.log(math.pow(10000,(i)/embed_dim)))))
        pe[k, i+1] = math.cos(k/(math.exp(math.log(math.pow(10000,(i)/embed_dim)))))
    pe = torch.unsqueeze(pe, 1).float()
    return pe