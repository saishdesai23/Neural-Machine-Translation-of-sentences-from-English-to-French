import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from importlib.machinery import SourceFileLoader
import sys
from positionalembedding import create_positional_embedding


class TransformerEncoder(nn.Module):
    
    def __init__(self, src_vocab, embedding_dim, num_heads,
        num_layers, dim_feedforward, max_len_src, device):
        super(TransformerEncoder, self).__init__()
        self.device = device
        """
        Args:
            src_vocab: Vocab_Lang, the source vocabulary
            embedding_dim: the dimension of the embedding (also the number of expected features for the input of the Transformer)
            num_heads: The number of attention heads
            num_layers: the number of Transformer Encoder layers
            dim_feedforward: the dimension of the feedforward network models in the Transformer
            max_len_src: maximum length of the source sentences
            device: the working device (you may need to map your postional embedding to this device)
        """
        self.src_vocab = src_vocab # Do not change
        src_vocab_size = len(src_vocab)

        # Create positional embedding matrix
        self.position_embedding = create_positional_embedding(max_len_src, embedding_dim).to(device)
        # self.position_embedding = torch.unsqueeze(self.position_embedding, 1)
        self.register_buffer('positional_embedding', self.position_embedding) # this informs the model that position_embedding is not a learnable parameter

        ### TODO ###
        # Initialize embedding layer
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.3)
        # Initialize a nn.TransformerEncoder model (you'll need to use embedding_dim,
        # num_layers, num_heads, & dim_feedforward here)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, device=self.device)
        self.transformer_model = nn.TransformerEncoder(encoder_layer, num_layers)
      
    def make_src_mask(self, src):
        """
        Args:
            src: [max_len, batch_size]
        Returns:
            Boolean matrix of size [batch_size, max_len] indicating which indices are padding
        """
        assert len(src.shape) == 2, 'src must have exactly 2 dimensions'
        src_mask = src.transpose(0, 1) == 0 # padding idx
        return src_mask.to(self.device) # [batch_size, max_src_len]

    def forward(self, x):
        """
        Args:
            x: [max_len, batch_size]
        Returns:
            output: [max_len, batch_size, embed_dim]
        Pseudo-code (note: x refers to the original input to this function throughout the pseudo-code):
        - Pass x through the word embedding
        - Add positional embedding to the word embedding, then apply dropout
        - Call make_src_mask(x) to compute a mask: this tells us which indexes in x
          are padding, which we want to ignore for the self-attention
        - Call the encoder, with src_key_padding_mask = src_mask
        """
        output = None

        ### TODO ###
        x1 = self.embedding(x)
        self.position_embedding = torch.permute(self.position_embedding, (1, 0, 2))
        if self.position_embedding.size()[1] < x1.size()[1]:
          d = x1.size()[1]- self.position_embedding.size()[1]
          # print(self.position_embedding.size())
          # print(torch.zeros(1, d, x1.size()[2]).size())
          self.position_embedding = torch.cat((self.position_embedding, torch.zeros(1, d, x1.size()[2]).to(self.device)), dim = 1)
        # print(self.position_embedding.size())
        # print(x.size())
        x1 = x1.to(self.device) + self.position_embedding[:,:x1.size(1), :]
        self.position_embedding = torch.permute(self.position_embedding, (1, 0, 2))
        x1 = self.dropout(x1)
        src_mask  = self.make_src_mask(x)
        # src_mask = src_mask.transpose(1,0)
        # print(src_mask)
        output = self.transformer_model(x1, src_key_padding_mask = src_mask)
        return output

