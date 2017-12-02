import numpy as np
import torch
from torch.autograd import Variable
from const import *

class DataLoader(object):
    def __init__(self, src_sents, max_len, batch_size, cuda=True):
        self.cuda = cuda
        self.sents_size = len(src_sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size

        self._batch_size = batch_size
        self._max_len = max_len
        self._enc_sents = np.asarray(src_sents)

        self._shuffle()
        self.gen_data()

    def gen_data(self):
        sents = np.copy(self._enc_sents)

        eos_tag = np.asarray([EOS]*self.sents_size).reshape((-1, 1))
        bos_tag = np.asarray([BOS]*self.sents_size).reshape((-1, 1))

        self._dec_sents = np.concatenate((bos_tag, sents), axis=-1)
        self._label = np.concatenate((sents, eos_tag), axis=-1)

    def _shuffle(self):
        indices = np.arange(self._enc_sents.shape[0])
        np.random.shuffle(indices)
        self._enc_sents = self._enc_sents[indices]

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

        enc_input = to_longest(self._enc_sents[_start: _start+_bsz])
        dec_input = to_longest(self._dec_sents[_start: _start+_bsz])
        label = to_longest(self._label[_start: _start+_bsz])
        return enc_input, dec_input, label

if __name__ == "__main__":
    data = torch.load("data/vae_nlg.pt")
    _data = DataLoader(
                     data['train'],
                     data["max_word_len"],
                     4)

    enc_input, dec_input, label = next(_data)

    print(enc_input)
    print(dec_input)
    print(label)

