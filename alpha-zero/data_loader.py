import numpy as np
import torch
import random

from const import *


def to_tensor(x, use_cuda=USECUDA, unsqueeze=False):
    x = torch.from_numpy(x).type(torch.Tensor)
    if use_cuda:
        x = x.cuda()

    if unsqueeze:
        x = x.unsqueeze(0)

    return x


def to_numpy(x, use_cuda=True):
    if use_cuda:
        return x.data.cpu().numpy().flatten()
    else:
        return x.data.numpy().flatten()


class DataLoader(object):
    def __init__(self, cuda, batch_size):
        self.cuda = cuda
        self.bsz = batch_size

    def __call__(self, datas):
        mini_batch = random.sample(datas, self.bsz)
        states, pi, rewards = [], [], []
        for s, p, r in mini_batch:
            states.append(s)
            pi.append(p)
            rewards.append(r)

        states = to_tensor(np.stack(states, axis=0), use_cuda=self.cuda)
        pi = to_tensor(np.stack(pi, axis=0), use_cuda=self.cuda)
        rewards = to_tensor(np.stack(rewards, axis=0), use_cuda=self.cuda)

        return states, pi, rewards.view(-1, 1)
