import numpy as np
import torch
from torch.autograd import Variable
import const


class DataLoader(object):
    def __init__(self, sents, chars, label, word_max_len, char_max_len, cuda=True,
                 batch_size=64, shuffle=True, evaluation=False):
        self.cuda = cuda
        self.sents_size = len(sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size
        self.evaluation = evaluation

        self._batch_size = batch_size
        self._word_max_len = word_max_len
        self._char_max_len = char_max_len
        self._sents = np.asarray(sents)
        self._chars = np.asarray(chars)
        self._label = np.asarray(label)

        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._sents.shape[0])
        np.random.shuffle(indices)
        self._sents = self._sents[indices]
        self._chars = self._chars[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def pad_to_longest(insts, is_char=False, word_max_len=None, char_max_len=None):
            if is_char:
                assert char_max_len is not None and word_max_len is not None

                temp_char = [const.PAD] * char_max_len
                temp_insts = np.array(
                    [inst + [temp_char] * (word_max_len - len(inst)) for inst in insts])
                inst_data = np.array(
                    [[word + [const.PAD] * (char_max_len - len(word)) for word in inst] for inst in temp_insts])
            else:
                max_len = max(len(inst) for inst in insts)
                inst_data = np.array(
                    [inst + [const.PAD] * (max_len - len(inst)) for inst in insts])

            if self.evaluation:
                with torch.no_grad():
                    inst_data_tensor = Variable(torch.from_numpy(inst_data))
            else:
                inst_data_tensor = Variable(torch.from_numpy(inst_data))

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step * self._batch_size
        _bsz = self._batch_size
        self._step += 1

        word = pad_to_longest(self._sents[_start:_start + _bsz])
        char = pad_to_longest(self._chars[_start:_start + _bsz],
                              True, word.size(1), self._char_max_len)
        label = pad_to_longest(self._label[_start:_start + _bsz])

        return word, char, label
