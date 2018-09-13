import numpy as np
import torch

from const import *


def to_tensor(x, use_cuda=USECUDA, unsqueeze=False):
    x = torch.from_numpy(x).type(torch.Tensor)
    if USECUDA:
        x = x.cuda()

    if unsqueeze:
        x = x.unsqueeze(0)

    return x


def to_numpy(x):
    return x.data.cpu().numpy().flatten()


class DataLoader(object):
    def __init__(self, cuda, batch_size):
        self.cuda = cuda
        self.bsz = batch_size

    def __call__(self, datas):
        self.data_size = len(datas)
        self.step = 0
        self.stop_step = self.data_size // self.bsz

        states, pi, rewards = [], [], []
        for s, p, r in datas:
            states.append(s)
            pi.append(p)
            rewards.append(r)

        states = np.stack(states, axis=0)
        pi = np.stack(pi, axis=0)
        rewards = np.asarray(rewards)

        indices = np.arange(rewards.shape[0])
        np.random.shuffle(indices)

        self.states = states[indices]
        self.pi = pi[indices]
        self.rewards = rewards[indices]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        self.step += 1

        start = self.step * self.bsz
        bsz = min(self.bsz, self.data_size - start)

        states = to_tensor(
            self.states[start:start + bsz], use_cuda=self.cuda)
        pi = to_tensor(
            self.pi[start:start + bsz], use_cuda=self.cuda)
        rewards = to_tensor(
            self.rewards[start:start + bsz], use_cuda=self.cuda)

        return states, pi, rewards.view(-1, 1)
