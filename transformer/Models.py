# -*- coding: utf-8 -*-
''' Define the Transformer model '''
import torch
import numpy as np
import math
import random
import torch.nn as nn
import transformer.Constants as Constants
from transformer.Embedding import PositionalEncoding
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.Utils import padding_info_mask, feature_info_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, input_dim, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_model=512, d_inner_hid=1024, dropout=0.1, emb_scale=1):

        super(Encoder, self).__init__()

        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.emb_scale = emb_scale

        # pos_embedding = position_encoding_init(self.n_max_seq, d_model)
        # self.position_enc = nn.Embedding.from_pretrained(pos_embedding, freeze=True)

        self.position_enc = PositionalEncoding(dropout, d_model, self.n_max_seq)
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head,
                         d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, inputs_data, inputs_pos, return_attns=False):

        # Position Encoding addition
        enc_input = self.input_proj(inputs_data)
        enc_input += self.position_enc(inputs_pos)

        enc_slf_attn_mask = padding_info_mask(inputs_pos, inputs_pos)

        enc_slf_attns = []
        enc_output = enc_input
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        return enc_output, enc_slf_attns


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, vocab_size, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_model=512, d_inner_hid=1024, dropout=0.1, emb_scale=1):

        super(Decoder, self).__init__()

        self.n_max_seq = n_max_seq
        self.output_dim = vocab_size
        self.d_model = d_model
        self.emb_scale = emb_scale

        # pos_embebding = position_encoding_init(
        #     self.n_max_seq, d_model)
        # self.position_enc = nn.Embedding.from_pretrained(
        #     pos_embebding, freeze=True)
        self.position_enc = PositionalEncoding(dropout, d_model, self.n_max_seq)

        self.tgt_word_emb = nn.Embedding(vocab_size, d_model, Constants.PAD)

        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head,
                         d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, outputs_data, outputs_pos, input_pos, enc_output, return_attns=False):

        # Word embedding look up
        dec_input = self.tgt_word_emb(outputs_data)

        # Position Encoding addition
        dec_input += self.position_enc(outputs_pos)

        dec_slf_attn_pad_mask = padding_info_mask(
            outputs_data, outputs_data)

        dec_slf_attn_sub_mask = feature_info_mask(outputs_data)
        dec_slf_attn_mask = torch.gt(
            dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = padding_info_mask(
            outputs_data, input_pos)

        dec_slf_attns, dec_enc_attns = [], []
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        return dec_output, dec_slf_attns, dec_enc_attns


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, config):
        super(Transformer, self).__init__()

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
        dec_output, dec_slf_attn, dec_enc_attn = self.decoder(targets, targets_pos, inputs_pos, enc_output,
                                                              return_attns)
        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit, (enc_slf_attn, dec_slf_attn, dec_enc_attn)
