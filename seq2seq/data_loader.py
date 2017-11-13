import numpy as np
import torch
from torch.autograd import Variable
import const

class DataLoader(object):
    def __init__(self, src_sents, tgt_sents, cuda, batch_size, shuffle=True, evaluation=False):
        self.cuda = cuda
        self.sents_size = len(src_sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size
        self.evaluation = evaluation
        self._batch_size = batch_size
        self._src_sents = np.asarray(src_sents)
        self._tgt_sents = np.asarray(tgt_sents)

        if shuffle:
            self.shuffle()

    def shuffle(self):
        index = np.arange(self._src_sents.shape[0])
        np.random.shuffle(index)
        self._src_sents = self._src_sents[index]
        self._tgt_sents = self._tgt_sents[index]

    def __iter__(self):
        return self

    def __next__(self):
        def pad_to_longest(insts):
            max_len = max(len(inst) for inst in insts)
            inst_data = np.array([inst + [const.PAD] * (max_len - len(inst)) for inst in insts])
            inst_position = np.array([[pos_i+1 if w_i != const.PAD else 0 for pos_i, w_i in enumerate(inst)] for inst in inst_data])

            inst_data_tensor = Variable(torch.from_numpy(inst_data), volatile=self.evaluation)
            inst_position_tensor = Variable(torch.from_numpy(inst_position), volatile=self.evaluation)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return (inst_data_tensor, inst_position_tensor)

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = self._batch_size
        self._step += 1

        src = pad_to_longest(self._src_sents[_start:_start+_bsz])
        tgt = pad_to_longest(self._tgt_sents[_start:_start+_bsz])

        return src, tgt
