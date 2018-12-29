''' Define the sublayers in encoder/decoder layer '''

import torch
import torch.nn as nn
import torch.nn.init as init

__author__ = "Zhengkun Tian"

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

        # initialization
        init.xavier_normal_(self.fc1.weight.data)
        init.xavier_normal_(self.fc2.weight.data)

    def forward(self, inputs):
        relu_output = self.dropout1(self.relu(self.fc1(inputs)))
        ffn_output = self.fc2(relu_output)
        output = self.dropout2(self.layernorm(inputs + ffn_output))
        return output

if __name__ == '__main__':
    ff = PositionwiseFeedForward(5, 10)
    inputs = torch.randn(2, 3, 5)
    output = ff(inputs)
    print(output.size())
