''' Define the Layers '''
import torch.nn as nn
from transformer.Attention import MultiHeadAttention
from transformer.SubLayers import PositionwiseFeedForward

__author__ = "Zhengkun Tian"

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.add_and_norm1 = nn.LayerNorm(d_model)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)
        self.add_and_norm2 = nn.LayerNorm(d_model)

    def forward(self, enc_input, slf_attn_mask=None):
        attn_output, slf_attn_weight = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        attn_norm_output = self.add_and_norm1(inputs + attn_output)
        pffn_output = self.pos_ffn(attn_norm_output)
        pffn_norm_output = self.add_and_norm2(attn_norm_output + pffn_output)
        return pffn_norm_output, slf_attn_weight


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.add_and_norm1 = nn.LayerNorm(d_model)
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.add_and_norm2 = nn.LayerNorm(d_model)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)
        self.add_and_norm3 = nn.LayerNorm(d_model)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        slf_attn_output, slf_attn_weight = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        slf_attn_norm_output = self.add_and_norm1(inputs + slf_attn_output)
        sre_attn_output, sre_attn_weight = self.enc_attn(
            slf_attn_norm_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        sre_attn_norm_output = self.add_and_norm2(slf_attn_norm_output + sre_attn_output)
        pffn_output = self.pos_ffn(sre_attn_norm_output)
        pffn_norm_output = self.add_and_norm3(sre_attn_norm_output+pffn_output)

        return pffn_norm_output, slf_attn_weight, sre_attn_weight
