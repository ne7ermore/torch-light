import numpy as np
import torch
from torch.autograd import Variable
import const


class DataLoader(object):
    def __init__(self, src_sents, tgt_sents, max_len, cuda, batch_size, evaluation=False):
        self.cuda = cuda
        self.max_len = max_len
        self.sents_size = len(src_sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size
        self.evaluation = evaluation
        self._batch_size = batch_size
        self._src_sents = np.asarray(src_sents)
        self._tgt_sents = np.asarray(tgt_sents)

    def __iter__(self):
        return self

    def __next__(self):
        def pad_to_longest(insts):
            inst_data = np.array(
                [inst + [const.PAD] * (self.max_len - len(inst)) for inst in insts])

            inst_data_tensor = Variable(torch.from_numpy(
                inst_data), volatile=self.evaluation)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step * self._batch_size
        _bsz = self._batch_size
        self._step += 1

        src = pad_to_longest(self._src_sents[_start:_start + _bsz])
        tgt = pad_to_longest(self._tgt_sents[_start:_start + _bsz])

        return src, tgt
