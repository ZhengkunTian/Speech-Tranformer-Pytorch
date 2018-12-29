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
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)

    def forward(self, inputs, slf_attn_mask=None):
        attn_output, slf_attn_weight = self.slf_attn(
            inputs, inputs, inputs, mask=slf_attn_mask)
        pffn_output = self.pos_ffn(attn_output)
        return pffn_output, slf_attn_weight


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)

    def forward(self,inputs, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        slf_attn_output, slf_attn_weight = self.slf_attn(
            inputs, inputs, inputs, mask=slf_attn_mask)
        sre_attn_output, sre_attn_weight = self.enc_attn(
            slf_attn_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        pffn_output = self.pos_ffn(sre_attn_output)

        return pffn_output, (slf_attn_weight, sre_attn_weight)
