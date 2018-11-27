import torch
import numpy as np
import random
import string

EOS = 3
BOS = 2
character = string.ascii_lowercase
encode = {character[i]: i+4 for i in range(len(character))}


def one_hot_embbedding(vocab_size):
    one_hot_mat = torch.eye(vocab_size)
    one_hot_mat[0][0] = 0
    embedding = torch.nn.Embedding.from_pretrained(one_hot_mat, freeze=True)
    return embedding


class random_character_loader(object):
    def __init__(self, batch_size, vocab_size, length, device):
        self.embedding = one_hot_embbedding(vocab_size)
        self.batch_size = batch_size
        self.length = length
        self.min_length = 6
        self.max_length = 10
        self.stop_iteration = False
        self.count = 0
        self.device = device

    def __iter__(self):
        self.count = 0
        self.stop_iteration = False
        return self

    def __next__(self):
        return self.next()

    def next(self):
        while not self.stop_iteration:
            if self.count >= self.length:
                self.stop_iteration = True
                raise StopIteration
                break
            else:
                return self.get_batch()

    def position_encoding(self, inputs):
        max_len = 0
        inputs_pos = []
        for item in inputs:
            length = len(item)
            max_len = max(length, max_len)
            inputs_pos.append(np.arange(length))
        inputs_pos = self.pad(inputs_pos)
        return inputs_pos

    def pad(self, inputs):
        dim = len(inputs[0].shape)
        max_len = max([inputs[i].shape[0] for i in range(len(inputs))])
        inst_data = []
        if dim == 1:
            for inst in inputs:
                pad_zeros_mat = np.zeros([1, max_len - inst.shape[0]])
                inst_data.append(np.column_stack([inst.reshape(1, -1), pad_zeros_mat]))
                padded_data = torch.LongTensor(np.row_stack(inst_data))
        elif dim == 2:
            feature_dim = inputs[0].shape[1]
            for inst in inputs:
                pad_zeros_mat = np.zeros([max_len-inst.shape[0], feature_dim])
                inst_data.append(np.row_stack([inst, pad_zeros_mat]).reshape(1, -1, feature_dim))
                padded_data = torch.FloatTensor(np.row_stack(inst_data))
        else:
            raise AssertionError(
                'Features in inputs list must be one vector or two dimension matrix! ')
        return padded_data

    def get_batch(self):
        self.count += 1
        inputs = []
        targets = []
        for i in range(self.batch_size):
            inst = self.random_inst()
            inputs.append(np.array(inst))
            targets.append(np.array([BOS]+inst+[EOS]))

        inputs_pos = self.position_encoding(inputs)
        targets_pos = self.position_encoding(targets)

        padded_features = self.embedding(self.pad(inputs))
        padded_targets = self.pad(targets)

        padded_features = padded_features.to(self.device)
        inputs_pos = inputs_pos.to(self.device)
        sos_targets = padded_targets[:, :-1].to(self.device)
        sos_targets_pos = targets_pos[:, :-1].to(self.device)
        targets_eos = padded_targets[:, 1:].to(self.device)

        return {
            'inputs': padded_features,
            'inputs_pos': inputs_pos,
            'sos_targets': sos_targets,
            'sos_targets_pos': sos_targets_pos,
            'targets_eos': targets_eos
        }

    def random_inst(self):
        length = random.randint(self.min_length, self.max_length)
        seq = random.sample(character, length)
        return [encode[i] for i in seq]


if __name__ == '__main__':
    loader = random_character_loader(4, 30, 20, torch.device('cpu'))
    for step, batch in enumerate(loader):
        print(batch['inputs'].requires_grad)
