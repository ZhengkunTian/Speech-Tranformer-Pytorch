import math
import torch
import torch.nn as nn
import numpy as np


def position_encoding_init(n_position, pos_enc_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / pos_enc_dim)
         for j in range(pos_enc_dim)]
        if pos != 0 else np.zeros(pos_enc_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=600):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, inputs, step=None):
        emb = inputs.mul(math.sqrt(self.dim))
        if step is None:
            emb = emb + self.pe[:emb.size(1)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


if __name__ == '__main__':
    embedding = PositionalEncoding(0.1, 10, 20)
    print(embedding.pe.size())
