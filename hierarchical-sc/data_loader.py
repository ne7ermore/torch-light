from collections import namedtuple

import torch
from torch.autograd import Variable
import numpy as np

import const


class DataLoader(object):
    def __init__(self,
                 original,
                 summary,
                 label,
                 max_ori_len,
                 max_sum_len,
                 use_cuda,
                 evaluation=False,
                 bsz=64,
                 shuffle=True):
        self.sents_size = len(original)
        self.step = 0
        self.stop_step = self.sents_size // bsz
        self.use_cuda = use_cuda
        self.evaluation = evaluation
        self.bsz = bsz
        self.max_ori_len = max_ori_len
        self.max_sum_len = max_sum_len
        self.original = np.asarray(original)
        self.summary = np.asarray(summary)
        self.label = label
        self.nt = namedtuple('dataloader', ['original', 'summary', 'label'])
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self.original.shape[0])
        np.random.shuffle(indices)
        self.original = self.original[indices]
        self.summary = self.summary[indices]
        self.label = self.label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def data_pad(sents, max_len):
            x = np.array([s + [const.PAD] * (max_len - len(s)) for s in sents])

            if self.evaluation:
                with torch.no_grad():
                    return Variable(torch.from_numpy(x))
            else:
                return Variable(torch.from_numpy(x))

        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        start = self.step * self.bsz
        bsz = min(self.bsz, self.sents_size - start)

        self.step += 1
        original = data_pad(self.original[start:start + bsz], self.max_ori_len)
        summary = data_pad(self.summary[start:start + bsz], self.max_sum_len)
        label = Variable(torch.from_numpy(self.label[start:start + bsz]))

        if self.use_cuda:
            original = original.cuda()
            summary = summary.cuda()
            label = label.cuda()

        return self.nt._make([original, summary, label])


if __name__ == '__main__':
    data = torch.load('./data/corpus')

    training_data = DataLoader(
        data['train']['original'],
        data['train']['summary'],
        data['train']['label'],
        data['max_ori_len'],
        data['max_sum_len'], True,
        bsz=128, evaluation=True, shuffle=False)

    dict = data["dict"]["dict"]
    idx2word = {v: k for k, v in dict.items()}
    dt = next(training_data)
