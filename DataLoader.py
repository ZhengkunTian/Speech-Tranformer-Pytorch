''' Data Loader class for training iteration '''
import random
import math
import numpy as np
import torch
import transformer.Constants as Constants
from processing import feature_reader
from processing import prepare_data
from processing import target_coder
from processing import target_normalizers

__author__ = "Zhengkun Tian"


class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_name, config, device, batch_size=4, context_width=0, frame_rate=10, shuffle=True, return_target=True, pad_to_max_len=False, return_numpy=False):

        feat_dir = config.get('directories', '%s_features' %
                              data_name) + '/' + config.get('dnn-features', 'name')

        textfile = config.get('directories', '%s_data' % data_name) + '/text'

        self.data_name = data_name

        self.conf = dict(config.items('dnn-features'))

        self.train = True if data_name == 'train' else False

        self.device = device

        with open(feat_dir + '/maxlength', 'r') as fid:
            self.max_input_length = int(fid.read())

        self.context_width = context_width

        self.frame_rate = frame_rate

        self.feature_reader = feature_reader.FeatureReader(
            feat_dir + '/feats.scp', feat_dir + '/cmvn.scp',
            feat_dir + '/utt2spk', context_width, self.max_input_length, frame_rate)

        self.num_utt = self.feature_reader.num_utt

        self._batch_size = batch_size
        self._n_batch = int(np.ceil(self.num_utt / self._batch_size))

        self._iter_count = 0

        self.target_dict, self._outputs_max_seq_lengths = self.read_target_file(
            textfile)

        self.target_coder = target_coder.ChinesePhoneEncoder(
            target_normalizers.SosEosNorm)

        self._need_shuffle = shuffle

        self._data_path = feat_dir

        self.return_target = return_target

        self.pad_to_max_len = pad_to_max_len

        self.return_numpy = return_numpy

        if self._need_shuffle:
            self.shuffle()

        self.get_info()

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return self.num_utt

    @property
    def vocab_size(self):
        '''the number of output labels'''

        return self.target_coder.num_labels

    @property
    def inputs_max_seq_lengths(self):
        max_seq_length = math.ceil(
            float(self.max_input_length) / (self.frame_rate / 10))
        return max_seq_length

    @property
    def outputs_max_seq_lengths(self):
        return self._outputs_max_seq_lengths

    @property
    def features_dim(self):
        dim = int(self.conf['nfilt'])
        include_energy = 1 if bool(self.conf['include_energy']) else 0
        dd = 2 if self.conf['dynamic'] == 'ddelta' else 0
        d = 1 if self.conf['dynamic'] == 'delta' else 0
        self._features_dim = (2 * self.context_width + 1) * \
            (dim + include_energy) * (1 + dd + d)

        return self._features_dim

    @property
    def tagter_coder(self):
        return tagter_coder

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        prepare_data.shuffle_examples(self._data_path)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def get_batch(self):
        '''
        Get a batch of features and targets.

        Returns:
            A pair containing:
                - The features: a list of feature matrices
                - The targets: a list of target vectors
        '''

        # set up the data lists.
        batch_inputs = []
        batch_targets = []
        # print(self._batch_size)
        while len(batch_inputs) < self._batch_size:
            # read utterance
            utt_id, utt_mat, _ = self.feature_reader.get_utt()

            # get transcription
            if utt_id in self.target_dict:
                targets = self.target_dict[utt_id]
                # print(utt_id)
                encoded_targets = self.target_coder.encode(targets)
                batch_inputs.append(utt_mat)
                batch_targets.append(encoded_targets)
            else:
                print('WARNING no targets for %s' % utt_id)

        return batch_inputs, batch_targets

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts, is_label=False, pad_to_max_len=False):
            ''' Pad the instance to the max seq length in batch '''
            # print(len(insts[0]))
            if not pad_to_max_len:
                max_len = max(inst.shape[0] for inst in insts)
            else:
                if is_label:
                    max_len = self.outputs_max_seq_lengths
                else:
                    max_len = self.inputs_max_seq_lengths

            dim = insts[0].shape[-1]
            insts_data = []
            insts_position = []
            for i in range(len(insts)):
                if not is_label:
                    pad_zeros_mat = np.zeros(
                        [max_len - insts[i].shape[0], dim], dtype=np.int16)
                    insts_data.append(np.row_stack([insts[i], pad_zeros_mat]))
                    insts_data[i] = insts_data[i][np.newaxis, :]

                else:
                    pad_zeros_mat = np.zeros(
                        [1, max_len - insts[i].shape[0]], dtype=np.int16)
                    insts_data.append(insts[i][np.newaxis, :])
                    insts_data[i] = np.column_stack(
                        [insts_data[i], pad_zeros_mat])

                inst_pos = np.arange(1, len(insts[i]) + 1, 1).reshape(1, -1)
                inst_pos_pad = np.zeros(
                    [1, max_len - inst_pos.shape[1]], dtype=np.int16)
                insts_position.append(
                    np.column_stack([inst_pos, inst_pos_pad]))

            inst_data = np.row_stack(insts_data)
            inst_position = np.row_stack(insts_position)

            if not self.return_numpy:
                if not is_label:
                    inst_data_tensor = torch.FloatTensor(
                        inst_data)
                else:
                    inst_data_tensor = torch.LongTensor(
                        inst_data)
                inst_position_tensor = torch.LongTensor(
                    inst_position)
            else:
                inst_data_tensor = inst_data
                inst_position_tensor = inst_position

            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            self._iter_count += 1

            batch_inputs, batch_targets = self.get_batch()
            inputs_data, inputs_pos = pad_to_longest(
                batch_inputs, is_label=False, pad_to_max_len=self.pad_to_max_len)

            if not self.return_target:
                return inputs_data, inputs_pos
            else:
                target_data, target_pos = pad_to_longest(
                    batch_targets, is_label=True, pad_to_max_len=self.pad_to_max_len)
                return (inputs_data, inputs_pos), (target_data, target_pos)

        else:
            self.feature_reader.reset_reader()

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()

    def read_target_file(self, target_path):
        '''
        read the file containing the text sequences

        Args:
            target_path: path to the text file

        Returns:
            A dictionary containing
                - Key: Utterance ID
                - Value: The target sequence as a string
        '''

        target_dict = {}
        max_len = 0
        with open(target_path, 'r', encoding='utf-8') as fid:
            for line in fid:
                splitline = line.strip().split(' ')
                tmp_len = max(len(splitline) + 2, max_len)
                max_len = tmp_len
                target_dict[splitline[0]] = ' '.join(splitline[1:])

        return target_dict, max_len

    def reset_loader(self):

        self.feature_reader.reset_reader()
        if self._need_shuffle:
            self.shuffle()

        self._iter_count = 0

    def get_info(self):
        print('**********************************')
        print('There are %d utts in %s set!' % (self.n_insts, self.data_name))
        print('The size of Vocab is %d' % self.vocab_size)
        print('The max length of input is %d' % self.inputs_max_seq_lengths)
        print('The max length of target is %d' % self.outputs_max_seq_lengths)
        print('The dimsion of feature is %d' % self.features_dim)
