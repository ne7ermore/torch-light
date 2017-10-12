import numpy as np
import torch
from torch.autograd import Variable
import const

class DataLoader(object):
    def __init__(self, src_sents, tgt_sents, label, max_src, max_tgt, cuda=True, batch_size=64, shuffle=True):
        assert len(src_sents) == len(tgt_sents)
        self.cuda = cuda
        self._batch_size = batch_size
        self._sents_size = len(src_sents)
        self._max_src = max_src
        self._max_tgt = max_tgt
        self._src_sents = np.asarray(src_sents)
        self._tgt_sents = np.asarray(tgt_sents)
        self._label = np.asarray(label)

        if shuffle:
            self.shuffle()

    def shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._tgt_sents = self._tgt_sents[indices]
        self._label = self._label[indices]

    def get_batch(self, i, evaluation=False):
        def pad_to_longest(insts, max_len):
            inst_data = np.array([inst + [const.PAD] * (max_len - len(inst)) for inst in insts])
            inst_data_tensor = Variable(torch.from_numpy(inst_data), volatile=evaluation)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor

        bsz = min(self._batch_size, self._sents_size-1-i)

        src = pad_to_longest(self._src_sents[i:i+bsz], self._max_src)
        tgt = pad_to_longest(self._tgt_sents[i:i+bsz], self._max_tgt)
        label = Variable(torch.from_numpy(self._label[i:i+bsz]), volatile=evaluation)
        if self.cuda:
                label = label.cuda()

        return src, tgt, label
