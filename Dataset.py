import codecs
import torch
import numpy as np
import tools.kaldi_io as kaldi_io
from transformer.Constants import EOS, BOS

class AudioDateset():
    def __init__(self, config, data_name):

        self.data_name = data_name
        self.name = config.name
        self.left_context_width = config.left_context_width
        self.right_context_width = config.right_context_width
        self.frame_rate = config.frame_rate
        self.apply_cmvn = config.apply_cmvn

        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length

        self.vocab = config.vocab
        self.text = config.__getattr__(data_name).text
        self.arkscp = config.__getattr__(data_name).arkscp

        if self.apply_cmvn:
            self.cmvnscp = config.__getattr__(data_name).cmvnscp
            self.cmvn_stats_dict = {}
            self.get_cmvn_dict()

        self.get_vocab_map()
        self.get_targets_dict()
        self.get_feats_list()
        self.vocab_size = len(self.unit2idx)

    def __getitem__(self, index):
        utt_id, feats_scp = self.feats_list[index]
        targets = np.array([BOS] + self.targets_dict[utt_id])
        groud_truth = np.array(self.targets_dict[utt_id] + [BOS])
        features = kaldi_io.read_mat(feats_scp)
        if self.apply_cmvn:
            spkid = self.extract_spk(utt_id)
            stats = self.cmvn_stats_dict[spkid]
            features = self.cmvn(features, stats)

        inputs_length = np.array(features.shape[0])
        targets_length = np.array(targets.shape[0])

        features = self.pad(features).astype(np.float32)
        targets = self.pad(targets).astype(np.int64).reshape(-1)
        groud_truth = self.pad(groud_truth).astype(np.int64).reshape(-1)

        return features, targets, inputs_length, targets_length, groud_truth

    def __len__(self):
        return self.lengths

    def pad(self, inputs):
        dim = len(inputs.shape)
        if dim == 1:
            pad_zeros_mat = np.zeros([1, self.max_target_length - inputs.shape[0]], dtype=np.int32)
            padded_inputs = np.column_stack([inputs.reshape(1, -1), pad_zeros_mat])
        elif dim == 2:
            feature_dim = inputs.shape[1]
            pad_zeros_mat = np.zeros([self.max_input_length - inputs.shape[0], feature_dim])
            padded_inputs = np.row_stack([inputs, pad_zeros_mat])
        else:
            raise AssertionError(
                'Features in inputs list must be one vector or two dimension matrix! ')
        return padded_inputs       

    def extract_spk(self, uttid):
        if self.name == 'aishell':
            return uttid[6:-5]
        elif self.name == 'timit':
            return uttid.split('_')[0]

    def get_feats_list(self):
        self.feats_list = []
        with open(self.arkscp, 'r') as fid:
            for line in fid:
                key, path = line.strip().split(' ')
                self.feats_list.append((key, path))
        self.lengths = len(self.feats_list)

    def get_cmvn_dict(self):
        cmvn_reader = kaldi_io.read_mat_scp(self.cmvnscp)
        for spkid, stats in cmvn_reader:
            self.cmvn_stats_dict[spkid] = stats

    def cmvn(self, mat, stats):
        mean = stats[0, :-1] / stats[0, -1]
        variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
        return np.divide(np.subtract(mat, mean), np.sqrt(variance))

    def get_vocab_map(self):
        self.unit2idx = {}
        with codecs.open(self.vocab, 'r', encoding='utf-8') as fid:
            idx = 0
            for line in fid:
                unit = line.strip().split(' ')[0]
                self.unit2idx[unit] = idx
                idx += 1

    def get_targets_dict(self):
        self.targets_dict = {}
        with codecs.open(self.text, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split(' ')
                utt_id = parts[0]
                labels = self.encode(parts[1:])
                self.targets_dict[utt_id] = labels

    def encode(self, seq):
        encoded_seq = []
        for unit in seq:
            if unit in self.unit2idx:
                encoded_seq.append(self.unit2idx[unit])
            else:
                encoded_seq.append(self.unit2idx['<unk>'])
        return encoded_seq

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
        if self.frame_rate != 10:
            interval = int(self.frame_rate / 10)
            temp_mat = [features[i]
                        for i in range(0, features.shape[0], interval)]
            subsampled_features = np.row_stack(temp_mat)
            return subsampled_features
        else:
            return features


if __name__ == '__main__':
    import yaml
    from transformer.utils import AttrDict
    import torch.utils.data
    configfile = open('config/timit.yaml')
    config = AttrDict(yaml.load(configfile))
    config.data.batch_size = 4

    dataset = AudioDateset(config.data, 'dev')
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    for step, batch in enumerate(loader):
        print(batch[0].size())
        print(batch[1].size())
        print(batch[2].cuda())
        print(batch[3].cuda())
        print(len(loader))
        break
