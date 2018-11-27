''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
import kaldi_io
import codecs
import time
from transformer.Constants import BOS_FLAG, EOS_FLAG, UNK_FLAG

__author__ = "Zhengkun Tian"


class DataLoader(object):
    ''' Load  '''

    def __init__(self, config, data_name, batch_size, vocab_size, left_context_width=0, right_context_width=0, frame_rate=10):

        self.config = config
        self.data = self.config.__getattr__(data_name)
        self.batch_size = batch_size
        self.left_context_width = left_context_width
        self.right_context_width = right_context_width
        self.frame_rate = frame_rate
        self.vocab_size = vocab_size
        self.stop_iteration = False
        self.get_vocab_map()
        self.get_targets_dict()

    def get_vocab_map(self):
        self.unit2idx = {}
        with codecs.open(self.config.vocab, 'r', encoding='utf-8') as fid:
            idx = 0
            for line in fid:
                unit = line.strip()
                self.unit2idx[unit] = idx
                idx += 1
        assert self.vocab_size == len(self.unit2idx)

    def get_targets_dict(self):
        self.targets_dict = {}
        with codecs.open(self.data.text, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split(' ')
                utt_id = parts[0]
                labels = self.encode(parts[1:])
                self.targets_dict[utt_id] = np.array(labels)

    def __iter__(self):
        self.reset()
        self.stop_iteration = False
        return self

    def __next__(self):
        return self.next()

    def next(self):
        while not self.stop_iteration:
            return self.get_batch()

    def encode(self, seq):
        seq.insert(0, BOS_FLAG)
        seq.append(EOS_FLAG)
        encoded_seq = []
        for unit in seq:
            if unit in self.unit2idx:
                encoded_seq.append(self.unit2idx[unit])
            else:
                encoded_seq.append(self.unit2idx[UNK_FLAG])
        return encoded_seq

    def get_batch(self):
        raise NotImplementedError

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

    def concat_frame(self, features):

        time_steps, features_dim = features.shape
        concated_features = np.zeros(
            shape=[time_steps, features_dim *
                   (1 + self.left_context_width + self.right_context_width)],
            dtype=np.float32)

        # middle part is just the uttarnce
        concated_features[:, self.left_context_width * features_dim:
                          (self.left_context_width + 1) * features_dim] = features

        for i in range(self.left_context_width):
            # add left context
            concated_features[i + 1:time_steps,
                              (self.left_context_width - i - 1) * features_dim:
                              (self.left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

        for i in range(self.right_context_width):
            # add right context
            concated_features[0:time_steps - i - 1,
                              (self.right_context_width + i + 1) * features_dim:
                              (self.right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

        return concated_features

    def subsampling(self, features):
        if self.frame_rate != 0:
            interval = int(self.frame_rate / 10)
            temp_mat = [features[i]
                        for i in range(0, features.shape[0], interval)]
            subsampled_features = np.row_stack(temp_mat)
            return subsampled_features
        else:
            return features

    def reset(self):
        raise NotImplementedError


class KaldiFeaturesLoader(DataLoader):
    def __init__(self, config, data_name, batch_size, vocab_size, apply_cmvn=False,
                 left_context_width=0, right_context_width=0, frame_rate=10, device=None, shuffle=True):
        super(KaldiFeaturesLoader, self).__init__(config, data_name, batch_size,
                                                  vocab_size, left_context_width, right_context_width, frame_rate)

        self.shuffle = shuffle
        self.apply_cmvn = apply_cmvn
        self.device = device
        self.data_list = []
        if self.apply_cmvn:
            self.cmvn_stats_dict = {}
            self.get_cmvn_dict()

    def extract_spk(self, uttid):
        # specific for aishell
        return uttid[6:-5]

    def get_cmvn_dict(self):
        cmvn_reader = kaldi_io.read_mat_scp(self.data.cmvnscp)
        for spkid, stats in cmvn_reader:
            self.cmvn_stats_dict[spkid] = stats

    def cmvn(self, mat, stats):
        mean = stats[0, :-1] / stats[0, -1]
        variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
        return np.divide(np.subtract(mat, mean), np.sqrt(variance))

    def get_batch(self):
        self.data_list = []
        while len(self.data_list) < self.batch_size:
            # check if there are remained data pairs in last epoch
            try:
                utt_id, mat = next(self.feature_reader)
            except StopIteration:
                self.stop_iteration = True
                break

            if self.apply_cmvn:
                spkid = self.extract_spk(utt_id)
                stats = self.cmvn_stats_dict[spkid]
                mat = self.cmvn(mat, stats)

            concated_mat = self.concat_frame(mat)
            mat = self.subsampling(concated_mat)

            labels = self.targets_dict[utt_id]
            self.data_list.append((mat, labels))

        if self.shuffle:
            random.shuffle(self.data_list)

        features, targets = [], []
        for (mat, labels) in self.data_list:
            features.append(mat)
            targets.append(labels)

        inputs_pos = self.position_encoding(features)
        targets_pos = self.position_encoding(targets)

        padded_features = self.pad(features)
        padded_targets = self.pad(targets)

        if self.device is not None:
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
            'targets_eos': targets_eos,
            'transcripts': targets
        }

    def reset(self):
        self.feature_reader = kaldi_io.read_mat_scp(self.data.arkscp)


def build_data_loader(config, data_name, device):
    if config.feature_source == 'kaldi':
        return KaldiFeaturesLoader(
            config=config.data,
            data_name=data_name,
            batch_size=config.data.batch_size,
            vocab_size=config.model.vocab_size,
            apply_cmvn=config.data.apply_cmvn,
            left_context_width=config.data.left_context_width,
            right_context_width=config.data.right_context_width,
            frame_rate=config.data.frame_rate,
            device=device,
            shuffle=True
        )
    else:
        raise NotImplementedError
