''' Define the sublayers in encoder/decoder layer '''

import torch
import torch.nn as nn
import torch.nn.init as init

__author__ = "Zhengkun Tian"

class PositionwiseFeedForward(nn.Module):

    def __init__(self, hidden_size, filter_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, filter_size, bias=True)
        self.fc2 = nn.Linear(filter_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # initialization
        init.xavier_normal_(self.fc1.weight.data)
        init.xavier_normal_(self.fc2.weight.data)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x

if __name__ == '__main__':
    ff = PositionwiseFeedForward(5, 10)
    inputs = torch.randn(2, 3, 5)
    output = ff(inputs)
    print(output.size())
