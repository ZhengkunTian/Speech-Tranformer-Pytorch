import torch
import argparse
import configparser
from transformer.Decode import Decode


class Evaluator(nn.Module):
    ''' A Evaluator is used to evaluate and test. '''

    def __init__(self, config):
        super(Evaluator, self).__init__()

        self.config = config
        self.encoder = Encoder(
            input_dim=config.feature_dim,
            n_max_seq=config.max_inputs_length,
            n_layers=config.num_enc_layer,
            n_head=config.n_heads,
            d_k=config.d_k,
            d_v=config.d_v,
            d_model=config.d_model,
            d_inner_hid=config.d_inner_hid,
            dropout=config.dropout,
            emb_scale=config.emb_scale)
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            n_max_seq=config.max_target_length,
            n_layers=config.num_dec_layer,
            n_head=config.n_heads,
            d_k=config.d_k,
            d_v=config.d_v,
            d_model=config.d_model,
            d_inner_hid=config.d_inner_hid,
            dropout=config.dropout,
            emb_scale=config.emb_scale)

        self.tgt_word_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, inputs, inputs_pos, targets=None, targets_pos=None, return_attns=False):

        enc_output, enc_slf_attn = self.encoder(inputs, inputs_pos, return_attns)

        for i in range(self.config.max_target_length):
            dec_output, dec_slf_attn, dec_enc_attn = self.decoder(targets, targets_pos, inputs_pos, enc_output,
                                                                  return_attns)
            seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit, (enc_slf_attn, dec_slf_attn, dec_enc_attn)
