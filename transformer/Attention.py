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

    def forward(self, Q, K, V, mask=None):
        """
        Q/K/V [batch_size, time_step, d_model]
        Args:
        Q: queue matrix
        K: key matrix
        V: value matrix
        QK^T:[batch_size, q_time_step, d_model]X[batch_size, d_model, k_time_step]
                        =[batch_size, q_time_step, k_time_step]
        """
        attn = torch.bmm(Q, K.transpose(1, 2)).div(self.scaled)

        if mask is not None:
            assert mask.size() == attn.size()
            attn.data.masked_fill_(mask, -float('inf'))

        attn_weights = self.softmax(attn)
        attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, V)

        return output, attn_weights


class SingleHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v):
        super(SingleHeadAttention, self).__init__()
        self.q_linear = nn.Linear(d_model, d_k, bias=False)
        self.k_linear = nn.Linear(d_model, d_k, bias=False)
        self.v_linear = nn.Linear(d_model, d_v, bias=False)
        self.attention = ScaledDotProductAttention(d_k)

        init.xavier_normal_(self.q_linear.weight)
        init.xavier_normal_(self.k_linear.weight)
        init.xavier_normal_(self.v_linear.weight)

    def forward(self, Q, K, V, mask=None):

        proj_q = self.q_linear(Q)
        proj_k = self.k_linear(K)
        proj_v = self.v_linear(V)
        output, attn_weights = self.attention(
            proj_q, proj_k, proj_v, mask=mask)
        return output, attn_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0
        assert d_v == int(d_model / n_head)
        assert d_k == int(d_model / n_head)

        self.attention = nn.ModuleList([SingleHeadAttention(
            d_model, d_k, d_v) for i in range(n_head)])

        self.Linear = nn.Linear(n_head * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        init.xavier_normal_(self.Linear.weight)

    def forward(self, Q, K, V, mask=None):
        results = []
        attns = []
        for i, single_attention in enumerate(self.attention):
            attention_out, attn_weights = single_attention(Q, K, V, mask=mask)
            results.append(attention_out)
            attns.append(attn_weights)

        concat = torch.cat(results, dim=-1)
        linear_output = self.Linear(concat)
        output = self.dropout(linear_output)
        return output, attns


if __name__ == '__main__':
    attention = ScaledDotProductAttention(6)
    inputs = torch.randn(1, 3, 6)
    mask = torch.ByteTensor([[[0, 0, 1], [0, 0, 1], [0, 0, 1]]])
    output, attn = attention(inputs, inputs, inputs, mask)
    single_attention = SingleHeadAttention(6, 3, 3)
    output, attn = single_attention(inputs, inputs, inputs, mask)
    print(inputs)
    print(output)
    print(attn)
    print(output.size())
    multi_attention = MultiHeadAttention(2, 6, 3, 3)
    output, attn = multi_attention(inputs, inputs, inputs, mask)
    print(output)
    print(output.size())
    print(attn)
