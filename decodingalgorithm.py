import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from positionalembedding import create_positional_embedding

def decode_transformer_model(encoder, decoder, src, max_decode_len, device):
    """
    Args:
        encoder: Your TransformerEncoder object
        decoder: Your TransformerDecoder object
        src: [max_src_length, batch_size] the source sentences you wish to translate
        max_decode_len: The maximum desired length (int) of your target translated sentences
        device: the device your torch tensors are on (you may need to call x.to(device) for some of your tensors)

    Returns:
        curr_output: [batch_size, max_decode_len] containing your predicted translated sentences
        curr_predictions: [batch_size, max_decode_len, trg_vocab_size] containing the (unnormalized) probabilities of each
            token in your vocabulary at each time step
    """
    # Initializing variables
    trg_vocab = decoder.trg_vocab
    batch_size = src.size(1)
    curr_output = torch.zeros((batch_size, max_decode_len))
    curr_predictions = torch.zeros((batch_size, max_decode_len, len(trg_vocab.idx2word)))
    enc_output = None

    # Decoding the start token for each example
    dec_input = torch.tensor([[trg_vocab.word2idx['<start>']]] * batch_size).transpose(0,1)
    curr_output[:, 0] = dec_input.squeeze(1)
    enc_output = encoder.forward(src)
    for t in range(1, max_decode_len):
      next_token = curr_output[:,:t].to(torch.long)
      decoder_output = decoder.forward(next_token.transpose(0,1), enc_output)
      curr_predictions[:,t,:] = decoder_output.permute(1, 0, 2)[:,-1,:]
      decoder_output = torch.argmax(curr_predictions[:,t,:], dim = -1)

      curr_output[:, t] = decoder_output
    return curr_output, curr_predictions, enc_output
    
def decode_rnn_model(encoder, decoder, src, max_decode_len, device):
    """
    Args:
        encoder: Your RnnEncoder object
        decoder: Your RnnDecoder object
        src: [max_src_length, batch_size] the source sentences you wish to translate
        max_decode_len: The maximum desired length (int) of your target translated sentences
        device: the device your torch tensors are on (you may need to call x.to(device) for some of your tensors)

    Returns:
        curr_output: [batch_size, max_decode_len] containing your predicted translated sentences
        curr_predictions: [batch_size, max_decode_len, trg_vocab_size] containing the (unnormalized) probabilities of each
            token in your vocabulary at each time step
    """
    # Initializing variables
    trg_vocab = decoder.trg_vocab
    batch_size = src.size(1)
    curr_output = torch.zeros((batch_size, max_decode_len))
    curr_predictions = torch.zeros((batch_size, max_decode_len, len(trg_vocab.idx2word)))

    # Decoding with the start token for each example
    dec_input = torch.tensor([[trg_vocab.word2idx['<start>']]] * batch_size)
    curr_output[:, 0] = dec_input.squeeze(1)
    encoder_output, encoder_hidden_state = encoder.forward(src)
    
    enc_to_dec = encoder_output
    decoder_hidden_state = encoder_hidden_state
    for t in range(1, max_decode_len):
      next_token = curr_output[:, t-1].type(torch.int64).to(device).unsqueeze(1)
      decoder_output, decoder_hidden_state, _= decoder.forward(next_token, decoder_hidden_state, enc_to_dec)
      curr_predictions[:,t,:] = decoder_output
      dec_input = torch.argmax(curr_predictions[:,t,:], dim = 1)
      curr_output[:, t] = dec_input
    
    
    return curr_output, curr_predictions