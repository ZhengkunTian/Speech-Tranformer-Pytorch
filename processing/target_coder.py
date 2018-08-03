'''@file target_coder.py
a file containing the target coders which can be used to encode and decode text,
alignments etc. '''

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np


class TargetCoder(object):
    '''an abstract class for a target coder which can encode and decode target
    sequences'''

    __metaclass__ = ABCMeta

    def __init__(self, target_normalizer):
        '''
        TargetCoder constructor

        Args:
            target_normalizer: a target normalizer function
        '''

        # save the normalizer
        self.target_normalizer = target_normalizer

        # create an alphabet of possible targets
        alphabet = self.create_alphabet()

        # create a lookup dictionary for fast encoding
        self.lookup = OrderedDict([(character, index) for index, character
                                   in enumerate(alphabet)])

    @abstractmethod
    def create_alphabet(self):
        '''create the alphabet for the coder'''

    def encode(self, targets):
        '''
        encode a target sequence

        Args:
            targets: a string containing the target sequence

        Returns:
            A numpy array containing the encoded targets
        '''

        # normalize the targets
        normalized_targets = self.target_normalizer(
            targets, self.lookup.keys())

        encoded_targets = []

        for target in normalized_targets.split(' '):
            if target not in self.lookup:
                encoded_targets.append(self.lookup['<unk>'])
            else:
                encoded_targets.append(self.lookup[target])

        return np.array(encoded_targets, dtype=np.uint32)

    def decode(self, encoded_targets):
        '''
        decode an encoded target sequence

        Args:
            encoded_targets: A numpy array containing the encoded targets

        Returns:
            A string containing the decoded target sequence
        '''

        targets = [self.lookup.keys()[encoded_target]
                   for encoded_target in encoded_targets]

        return ' '.join(targets)

    @property
    def num_labels(self):
        '''the number of possible labels'''

        return len(self.lookup)


class TextCoder(TargetCoder):
    '''a coder for text'''

    def create_alphabet(self):
        '''create the alphabet of characters'''

        alphabet = []
        # end of sentence token
        alphabet.append('<eos>')
        # start of sentence token
        alphabet.append('<sos>')
        # space
        alphabet.append('<space>')
        # comma
        alphabet.append(',')
        # period
        alphabet.append('.')
        # apostrophy
        alphabet.append('\'')
        # hyphen
        alphabet.append('-')
        # question mark
        alphabet.append('?')
        # unknown character
        alphabet.append('<unk>')
        # letters in the alphabet
        for letter in range(ord('a'), ord('z') + 1):
            alphabet.append(chr(letter))

        return alphabet


class AlignmentCoder(TargetCoder):
    '''a coder for state alignments'''

    def __init__(self, target_normalizer, num_targets):
        '''
        AlignmentCoder constructor

        Args:
            target_normalizer: a target normalizer function
            num_targets: total number of targets
        '''

        self.num_targets = num_targets
        super(AlignmentCoder, self).__init__(target_normalizer)

    def create_alphabet(self):
        '''
        create the alphabet of alignment targets
        '''

        alphabet = [str(target) for target in range(self.num_targets)]

        return alphabet


class PhonemeEncoder(TargetCoder):
    """ Sets up a 39 element foldet phoneme alphabet."""

    def create_alphabet(self):
        """
        Create an alphabet of folded phonemes, according to
        "Speaker-Independent Phone Recognition Using Hidden Markov Models."
        """

        alphabet = ['sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh',
                    'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
                    'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh',
                    't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
        return alphabet


class LasPhonemeEncoder(TargetCoder):
    """ Sets up a 39 element foldet phoneme alphabet."""

    def create_alphabet(self):
        """
        Create an alphabet of folded phonemes, according to
        "Speaker-Independent Phone Recognition Using Hidden Markov Models."
        It is very important to have the eos tokan at poistion one, because the las
        code assumes it is there!
        """

        alphabet = ['<eos>', '<sos>', 'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh',
                    'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
                    'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh',
                    't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
        return alphabet


class ChinesePhoneEncoder(TargetCoder):
    """ Sets up a 221 element foldet phoneme alphabet."""

    def create_alphabet(self):

        PHONES = "<blank> <unk> <s> </s> <eps> a1 a2 a3 a4 a5 aa ai1 ai2 ai3 ai4 ai5 an1 an2 an3 an4 an5 ang1 ang2 ang3 ang4 ang5 ao1 ao2 ao3 ao4 ao5 b c ch d e1 e2 e3 e4 e5 ee ei1 ei2 ei3 ei4 ei5 en1 en2 en3 en4 en5 eng1 eng2 eng3 eng4 eng5 er2 er3 er4 er5 f g h i1 i2 i3 i4 i5 ia1 ia2 ia3 ia4 ia5 ian1 ian2 ian3 ian4 ian5 iang1 iang2 iang3 iang4 iang5 iao1 iao2 iao3 iao4 iao5 ie1 ie2 ie3 ie4 ie5 ii in1 in2 in3 in4 in5 ing1 ing2 ing3 ing4 ing5 iong1 iong2 iong3 iong4 iong5 iu1 iu2 iu3 iu4 iu5 ix1 ix2 ix3 ix4 ix5 iy1 iy2 iy3 iy4 iy5 iz4 iz5 j k l m n o1 o2 o3 o4 o5 ong1 ong2 ong3 ong4 ong5 oo ou1 ou2 ou3 ou4 ou5 p q r s sh sil t u1 u2 u3 u4 u5 ua1 ua2 ua3 ua4 ua5 uai1 uai2 uai3 uai4 uai5 uan1 uan2 uan3 uan4 uan5 uang1 uang2 uang3 uang4 uang5 ueng1 ueng3 ueng4 ueng5 ui1 ui2 ui3 ui4 ui5 un1 un2 un3 un4 un5 uo1 uo2 uo3 uo4 uo5 uu v1 v2 v3 v4 v5 van1 van2 van3 van4 van5 ve1 ve2 ve3 ve4 ve5 vn1 vn2 vn3 vn4 vn5 vv x z zh"
        alphabet = PHONES.split()
        return alphabet
