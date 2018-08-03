# -*- coding: utf-8 -*-
''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Modules import BottleLinear as Linear
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec)
         for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    # 遮掉填充的信息
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    subsequent_mask = subsequent_mask.to(device)
    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, input_dim, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_model, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(
            n_position, d_model)

        self.input_proj = Linear(input_dim, d_model)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head,
                         d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, inputs_data, inputs_pos, return_attns=False):

        # Position Encoding addition
        inputs_data = self.input_proj(inputs_data)
        enc_input = inputs_data + self.position_enc(inputs_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(inputs_pos, inputs_pos)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, output_dim, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, output_dim, padding_idx=Constants.PAD)

        self.position_enc.weight.data = position_encoding_init(
            n_position, output_dim)

        # self.tgt_word_emb = nn.Embedding(
        #     output_dim, output_dim, padding_idx=Constants.PAD)

        self.tgt_word_emb = nn.Embedding(
            output_dim, output_dim, padding_idx=Constants.PAD)

        self.dropout = nn.Dropout(dropout)

        self.input_proj = Linear(output_dim, d_model)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head,
                         d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, outputs_data, outputs_pos, input_pos, enc_output, return_attns=False):
        # Word embedding look up
        dec_input = self.tgt_word_emb(outputs_data)

        dec_input = self.input_proj(dec_input)

        # Position Encoding addition
        dec_input += self.position_enc(outputs_pos)

        # Decode
        # mask 去掉填充的信息
        dec_slf_attn_pad_mask = get_attn_padding_mask(
            outputs_data, outputs_data)
        # mask 去掉每一步未来的信息
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(outputs_data)
        dec_slf_attn_mask = torch.gt(
            dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        # mask去掉在输入序列长度之后的信息
        dec_enc_attn_pad_mask = get_attn_padding_mask(outputs_data, input_pos)

        if return_attns:
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

        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, input_dim, output_dim, n_inputs_max_seq, n_outputs_max_seq, n_layers=6,
            n_head=8, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            dropout=0.1):

        super(Transformer, self).__init__()
        self.encoder = Encoder(
            input_dim, n_inputs_max_seq, n_layers=n_layers, n_head=n_head,
            d_model=d_model, d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder = Decoder(
            output_dim, n_outputs_max_seq, n_layers=n_layers, n_head=n_head,
            d_model=d_model, d_inner_hid=d_inner_hid, dropout=dropout)
        self.tgt_word_proj = Linear(d_model, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # assert d_model == d_word_vec, \
        #     'To facilitate the residual connections, \
        #  the dimensions of all module output shall be the same.'

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(
            map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(
            map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, inputs, targets):
        src_seq, src_pos = inputs
        tgt_seq, tgt_pos = targets

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))
