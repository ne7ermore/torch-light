import numpy as np
import torch
from torch.autograd import Variable
import const

class DataLoader(object):
    def __init__(self, src_sents, max_len, batch_size, cuda=True):
        self.cuda = cuda
        self.sents_size = len(src_sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size

        self._batch_size = batch_size
        self._max_len = max_len

        self.gen_data(src_sents)

    def gen_data(self, src_sents):
        src_sents = np.asarray(src_sents)
        self._src_sents = src_sents[:, :-1]
        self._label = src_sents[:, 1:]

    def _shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def to_longest(insts):
            inst_data_tensor = Variable(torch.from_numpy(insts))
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = self._batch_size
        self._step += 1
        data = to_longest(self._src_sents[_start: _start+_bsz])
        label = to_longest(self._label[_start: _start+_bsz])
        return data, label.contiguous().view(-1)

if __name__ == "__main__":
    data = torch.load("data/ch_pro_nlg.pt")
    _data = DataLoader(
                     data['train'],
                     data["max_word_len"],
                     64)
    d = {v: k for k, v in data['dict']['src'].items()}

    print([d[w] for s in _data._src_sents for w in s])
    print([d[w] for s in _data._label for w in s])

