import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RnnDecoder(nn.Module):
    def __init__(self, trg_vocab, embedding_dim, hidden_units):
        super(RnnDecoder, self).__init__()
        """
        Args:
            trg_vocab: Vocab_Lang, the target vocabulary
            embedding_dim: The dimension of the embedding
            hidden_units: The number of features in the GRU hidden state
        """
        self.trg_vocab = trg_vocab # Do not change
        vocab_size = len(trg_vocab)

        
        # Initializing embedding layer
        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Initialize layers to compute attention score
        self.W_1 = nn.Linear(hidden_units, hidden_units)
        self.W_2 = nn.Linear(hidden_units, hidden_units)
        self.v = nn.Linear(hidden_units, 1)
        
        # Initializing a single directional GRU with 1 layer and batch_first=True
        self.decoder_gru = nn.GRU(embedding_dim + hidden_units, hidden_units, num_layers = 1, batch_first=False)
        # Initializing fully connected layer
        self.fc = nn.Linear(hidden_units, vocab_size)

    def compute_attention(self, dec_hs, enc_output):
        '''
        This function computes the context vector and attention weights.

        Args:
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            enc_output: Encoder outputs; [max_len_src, batch_size, hidden_units]

        Returns:
            context_vector: Context vector, according to formula; [batch_size, hidden_units]
            attention_weights: The attention weights you have calculated; [batch_size, max_len_src, 1]

        '''      
        context_vector, attention_weights = None, None
        
        
        enc_output = torch.permute(enc_output, (1, 0 ,2))
        dec_hs = torch.permute(dec_hs, (1, 0 ,2))
        weight = self.W_1(dec_hs) + self.W_2(enc_output)
        attention_weights = torch.nn.functional.softmax(self.v(torch.tanh(weight)), dim = 1)
        context_vector = torch.sum(torch.mul(attention_weights, enc_output), dim = 1)
        return context_vector, attention_weights

    def forward(self, x, dec_hs, enc_output):
        '''
        This function runs the decoder for a **single** time step.

        Args:
            x: Input token; [batch_size, 1]
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            enc_output: Encoder outputs; [max_len_src, batch_size, hidden_units]

        Returns:
            fc_out: (Unnormalized) output distribution [batch_size, vocab_size]
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            attention_weights: The attention weights you have learned; [batch_size, max_len_src, 1]

        '''
        fc_out, attention_weights = None, None

        
        context_vector, attention_weights = self.compute_attention(dec_hs, enc_output)
        x = self.decoder_embedding(x)
        context_vector = torch.unsqueeze(context_vector, 1)
        x = torch.cat((x, context_vector), dim = 2)
        x = torch.permute(x, (1, 0 ,2))
        output, hidden_state = self.decoder_gru(x, dec_hs)
        output = torch.squeeze(output, dim = 0)
        fc_out = self.fc(output)
        
        


        return fc_out, dec_hs, attention_weights
