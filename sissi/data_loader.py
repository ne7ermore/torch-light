from collections import namedtuple
import os
import random

import torch
from torch.autograd import Variable
import numpy as np

from util import *

styles = [
    'data/beeth',
    'data/brahms',
    'data/burgm',
    'data/chopin',
    'data/haydn',
    'data/mozart'
]


def load_midi(fname, cache_dir="cache"):
    cache_path = os.path.join(cache_dir, fname + ".npy")
    try:
        stateMatrix = np.load(cache_path)
    except Exception as e:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        stateMatrix = np.asarray(midiToNoteStateMatrix(fname))
        np.save(cache_path, stateMatrix)

    return stateMatrix


class DataLoader(object):
    def __init__(self,
                 bsz=10,
                 use_cuda=True,
                 batch_len=16 * 8):

        self.bsz = bsz
        self.use_cuda = use_cuda
        self.batch_len = batch_len

        self._init_data()

    def _init_data(self):
        pieces = {}

        for style in styles:
            for f in os.listdir(style):
                stateMatrix = load_midi(os.path.join(style, f))
                if len(stateMatrix) >= self.batch_len:
                    pieces[f] = stateMatrix
                    # print("loaded {}".format(f))

        self.pieces = pieces

    def getPieceBatch(self):
        seg_ins, seg_outs = [], []
        for _ in range(self.bsz):
            piece_output = random.choice(list(self.pieces.values()))
            start = random.randrange(
                0, len(piece_output) - self.batch_len, 10)
            seg_out = piece_output[start:start + self.batch_len]
            seg_in = noteStateMatrixToInputForm(seg_out)

            seg_ins.append(seg_in)
            seg_outs.append(seg_out)

        return self._l2V(seg_ins), self._l2V(seg_outs)

    def _l2V(self, ins):
        out = np.stack(ins, axis=0)
        out = torch.from_numpy(out.astype(np.float32))
        if self.use_cuda:
            out = out.cuda()
        return out


if __name__ == "__main__":
    d = DataLoader()
    seg_ins, seg_outs = d.getPieceBatch()
    print(seg_ins.shape)
    print(seg_outs.shape)
    seg_ins, seg_outs = seg_ins[0, 0], seg_outs[0, 0]
    print(seg_ins.shape)
    print(seg_outs)

    # (nd, bd, sd, nt), l = next(d)
    # print(l.shape)
    # print(nd.shape)
    # print(bd.shape)
    # print(sd.shape)
    # print(nt.shape)
