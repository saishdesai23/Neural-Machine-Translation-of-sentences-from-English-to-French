import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from positionalembedding import create_positional_embedding

class TransformerDecoder(nn.Module):
    def __init__(self, trg_vocab, embedding_dim, num_heads,
        num_layers, dim_feedforward, max_len_trg, device):
        super(TransformerDecoder, self).__init__()
        self.device = device
        """
        Args:
            trg_vocab: Vocab_Lang, the target vocabulary
            embedding_dim: the dimension of the embedding (also the number of expected features for the input of the Transformer)
            num_heads: The number of attention heads
            num_layers: the number of Transformer Decoder layers
            dim_feedforward: the dimension of the feedforward network models in the Transformer
            max_len_trg: maximum length of the target sentences
            device: the working device (you may need to map your postional embedding to this device)
        """
        self.trg_vocab = trg_vocab # Do not change
        trg_vocab_size = len(trg_vocab)
        # Creating positional embedding matrix
        self.position_embedding = create_positional_embedding(max_len_trg, embedding_dim).to(self.device)
        self.register_buffer('positional_embedding', self.position_embedding)


        # Initializing embedding layer
        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim, device=self.device)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.3)
        # Initializing a nn.TransformerDecoder model
        decoder_layers = nn.TransformerDecoderLayer(embedding_dim, num_heads, dim_feedforward, device=self.device)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc1 = nn.Linear(embedding_dim, trg_vocab_size, device=self.device)


    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, dec_in, enc_out):
        """
        Args:
            dec_in: [sequence length, batch_size]
            enc_out: [max_len, batch_size, embed_dim]
        Returns:
            output: [sequence length, batch_size, trg_vocab_size]
        """
        output = None
        
        x1 = self.embedding(dec_in.to(self.device))
        x1.to(self.device)
        self.position_embedding = torch.permute(self.position_embedding, (1, 0, 2))
        if self.position_embedding.size()[1] < x1.size()[1]:
          d = x1.size()[1]- self.position_embedding.size()[1]
          self.position_embedding = torch.cat((self.position_embedding, torch.zeros(1, d, x1.size()[2]).to(self.device)), dim = 1)
        x1 = x1.to(self.device) + self.position_embedding[:,:x1.size(1), :].to(self.device)
        self.position_embedding = torch.permute(self.position_embedding, (1, 0, 2))
        trg_mask = self.generate_square_subsequent_mask(dec_in.size(0))
        tgt = self.transformer_decoder(x1.to(self.device), enc_out.to(self.device), trg_mask.to(self.device))
        output = self.fc1(tgt)
        return output