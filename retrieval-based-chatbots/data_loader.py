from collections import namedtuple

import torch
from torch.autograd import Variable
import numpy as np

from const import *


def reps_pad(responses, max_len, evaluation):
    x = np.array([resp + [PAD] * (max_len - len(resp)) for resp in responses])
    if evaluation:
        with torch.no_grad():
            x = Variable(torch.from_numpy(x))
    else:
        x = Variable(torch.from_numpy(x))

    return x


def uttes_pad(utterances, max_cont_len, max_utte_len, evaluation):
    pad_utte = [[PAD] * max_utte_len]
    utterances = [[u + [PAD] * (max_utte_len - len(u))
                   for u in utte] for utte in utterances]
    utterances = [pad_utte * (max_cont_len - len(utte)) +
                  utte for utte in utterances]

    x = np.array(utterances)
    if evaluation:
        with torch.no_grad():
            x = Variable(torch.from_numpy(x))
    else:
        x = Variable(torch.from_numpy(x))

    return x


class DataLoader(object):
    def __init__(self,
                 utterances,
                 responses,
                 labels,
                 max_cont_len,
                 max_utte_len,
                 use_cuda,
                 evaluation=False,
                 bsz=64,
                 shuffle=True):
        self.sents_size = len(utterances)
        self.step = 0
        self.stop_step = self.sents_size // bsz
        self.bsz = bsz
        self.use_cuda = use_cuda
        self.evaluation = evaluation
        self.max_cont_len = max_cont_len
        self.max_utte_len = max_utte_len
        self.utterances = np.asarray(utterances)
        self.responses = np.asarray(responses)
        self.labels = np.asarray(labels)
        self.nt = namedtuple(
            'dataloader', ['utterances', 'responses', 'labels'])

        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self.utterances.shape[0])
        np.random.shuffle(indices)
        self.utterances = self.utterances[indices]
        self.responses = self.responses[indices]
        self.labels = self.labels[indices]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        start = self.step * self.bsz
        bsz = min(self.bsz, self.sents_size - start)
        self.step += 1

        utterances = uttes_pad(
            self.utterances[start:start + bsz], self.max_cont_len, self.max_utte_len, self.evaluation)
        responses = reps_pad(
            self.responses[start:start + bsz], self.max_utte_len, self.evaluation)
        labels = Variable(torch.from_numpy(self.labels[start:start + bsz]))

        if self.use_cuda:
            utterances = utterances.cuda()
            responses = responses.cuda()
            labels = labels.cuda()

        return self.nt._make([utterances, responses, labels])


if __name__ == '__main__':
    data = torch.load('./data/corpus')

    training_data = DataLoader(
        data['train']['utterances'],
        data['train']['responses'],
        data['train']['labels'],
        data['max_cont_len'],
        data['max_utte_len'],
        True, bsz=4, shuffle=False, evaluation=True)

    dict = data["dict"]["dict"]
    idx2word = {v: k for k, v in dict.items()}
    u, r, l = next(training_data)
    print(u)
    print(r)
    print(l)
