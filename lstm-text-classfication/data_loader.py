import numpy as np
import torch
from torch.autograd import Variable
import const

class DataLoader(object):
    def __init__(self, src_sents, label, max_len, cuda=True,
                batch_size=64, shuffle=True, evaluation=False):
        self.cuda = cuda
        self.sents_size = len(src_sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size
        self.evaluation = evaluation

        self._batch_size = batch_size
        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._label = np.asarray(label)
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def pad_to_longest(insts, max_len):
            inst_data = np.array([inst + [const.PAD] * (max_len - len(inst)) for inst in insts])

            inst_data_tensor = Variable(torch.from_numpy(inst_data), volatile=self.evaluation)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = self._batch_size
        self._step += 1
        data = pad_to_longest(self._src_sents[_start:_start+_bsz], self._max_len)
        label = Variable(torch.from_numpy(self._label[_start:_start+_bsz]),
                    volatile=self.evaluation)
        if self.cuda:
            label = label.cuda()

        return data, label
