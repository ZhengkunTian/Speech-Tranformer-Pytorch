import math
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=600):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, inputs_length, step=None):

        batch_size = inputs_length.size(0)
        time_steps = inputs_length.max().item()
        if step is None:
            pos_enc = self.pe[:, :time_steps].repeat(batch_size, 1, 1)
        else:
            pos_enc = self.pe[:, step].repeat(batch_size, 1, 1)
        return pos_enc


if __name__ == '__main__':
    embedding = PositionalEncoding(0.1, 10, 20)
    print(embedding.pe.size())
    inputs = torch.randn([4, 3, 10])
    print(embedding(inputs))
