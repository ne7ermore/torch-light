import numpy as np
import torch
from torch.autograd import Variable
import const

class DataLoader(object):
    def __init__(self, src_sents, tgt_sents, cuda=True, batch_size=64, shuffle=True):
        assert len(src_sents) == len(tgt_sents)
        self.cuda = cuda
        self._batch_size = batch_size
        self._sents_size = len(src_sents)
        self._src_sents = np.asarray(src_sents)
        self._tgt_sents = np.asarray(tgt_sents)

        if shuffle:
            self.shuffle()

    def shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._tgt_sents = self._tgt_sents[indices]

    def get_batch(self, i, evaluation=False):
        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''
            max_len = max(len(inst) for inst in insts)
            inst_data = np.array([inst + [const.PAD] * (max_len - len(inst)) for inst in insts])
            inst_position = np.array([[pos_i+1 if w_i != const.PAD else 0 for pos_i, w_i in enumerate(inst)] for inst in inst_data])

            inst_data_tensor = Variable(torch.from_numpy(inst_data), volatile=evaluation)
            inst_position_tensor = Variable(torch.from_numpy(inst_position), volatile=evaluation)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return (inst_data_tensor, inst_position_tensor)

        bsz = min(self._batch_size, self._sents_size-1-i)

        src = pad_to_longest(self._src_sents[i:i+bsz])
        tgt = pad_to_longest(self._tgt_sents[i:i+bsz])

        return src, tgt
