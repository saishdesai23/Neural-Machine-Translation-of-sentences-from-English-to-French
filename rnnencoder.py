import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RnnEncoder(nn.Module):
    def __init__(self, src_vocab, embedding_dim, hidden_units):
        super(RnnEncoder, self).__init__()
        """
        Args:
            src_vocab: Vocab_Lang, the source vocabulary
            embedding_dim: the dimension of the embedding
            hidden_units: The number of features in the GRU hidden state
        """
        self.src_vocab = src_vocab # Do not change
        vocab_size = len(src_vocab)

        self.embedding_dim = embedding_dim
        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Initialize a single directional GRU with 1 layer and batch_first=False
        self.gru = nn.GRU(embedding_dim, hidden_units, batch_first=False)
        

    def forward(self, x):
        """
        Args:
            x: source texts, [max_len, batch_size]

        Returns:
            output: [max_len, batch_size, hidden_units]
            hidden_state: [1, batch_size, hidden_units] 
        
        Pseudo-code:
        - Pass x through an embedding layer and pass the results through the recurrent net
        - Return output and hidden states from the recurrent net
        """
        output, hidden_state = None, None

        x = self.embedding(x)
        output, hidden_state = self.gru(x, hidden_state)
        return output, hidden_state
