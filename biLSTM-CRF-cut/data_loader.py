import numpy as np
import torch
from torch.autograd import Variable
import const

class DataLoader(object):
    def __init__(self, src_sents, label, max_len, cuda=True, batch_size=64, shuffle=True):
        self.cuda = cuda
        self.sents_size = len(src_sents)

        self._batch_size = batch_size
        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._label = np.asarray(label)
        if shuffle:
            self.shuffle()

    def shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._label = self._label[indices]

    def get_batch(self, i, evaluation=False):
        def pad_to_longest(insts, max_len):
            inst_data = np.array([inst + [const.X] * (max_len - len(inst)) for inst in insts])

            inst_data_tensor = Variable(torch.from_numpy(inst_data), volatile=evaluation)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor

        bsz = min(self._batch_size, self.sents_size-1-i)

        src = pad_to_longest(self._src_sents[i:i+bsz], self._max_len)
        label = pad_to_longest(self._label[i:i+bsz], self._max_len)
        return src, label.view(-1)
