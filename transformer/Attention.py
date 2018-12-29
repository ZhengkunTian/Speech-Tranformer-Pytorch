import math
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

__author__ = "Zhengkun Tian"

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.scaled = math.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Q/K/V [batch_size, time_step, d_model]
        Args:
        Q: queue matrix
        K: key matrix
        V: value matrix
        QK^T:[batch_size, q_time_step, d_model]X[batch_size, d_model, k_time_step]
                        =[batch_size, q_time_step, k_time_step]
        """
        attn = torch.bmm(q, k.transpose(1, 2)).div(self.scaled)

        if mask is not None:
            assert mask.size() == attn.size()
            attn.data.masked_fill_(mask, -float('inf'))

        attn_weights = self.softmax(attn)
        attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0
        assert d_v == int(d_model / n_head)
        assert d_k == int(d_model / n_head)

        self.d_model = d_model
        self.n_head =  n_head
        self.d_k = d_k
        self.d_v = d_v
        self.scaled = math.sqrt(d_k)

        self.linear_q = nn.Linear(d_model, n_head * d_k)
        self.linear_k = nn.Linear(d_model, n_head * d_k)
        self.linear_v = nn.Linear(d_model, n_head * d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)

        def shape(x):
            return x.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_k)

        query = self.linear_q(q)
        key = self.linear_k(k)
        value = self.linear_v(v)

        query = shape(query)
        key = shape(key)
        value = shape(value)

        scores = torch.matmul(query, key.transpose(2, 3)).div(self.scaled)

        if mask is not None:
            # [b_size x len_q x len_k]
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            scores.masked_fill_(mask, -float('inf'))
        
        attns = self.dropout(self.softmax(scores))
        context = unshape(torch.matmul(attns, value))

        output = self.output_linear(context)

        norm_output = self.layernorm(output + v)

        return norm_output, attns


if __name__ == '__main__':
    inputs = torch.randn(2, 3, 6)
    mask = torch.ByteTensor([[[0, 0, 0], [0, 0, 1], [0, 1, 1]], [[0, 0, 0], [0, 0, 1], [0, 1, 1]]])
    multi_attention = MultiHeadAttention(2, 6, 3, 3)
    multi_attention.eval()
    output, attn = multi_attention(inputs, inputs, inputs, mask)
    print(output)
    print(output.size())
    print(attn)
    print(attn.size())
