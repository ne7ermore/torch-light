import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from const import *


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, props, tgt):
        tgt_props = props.gather(2, tgt.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        return -(tgt_props * mask).sum() / mask.sum()


class SelfCriticCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, props, s_words, tgt, advantage):
        advantage = (advantage - advantage.mean()) / \
            advantage.std().clamp(min=1e-8)
        s_props = props.gather(2, s_words.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        advantage = advantage.unsqueeze(1).repeat(1, mask.size(1))
        advantage = advantage.detach()

        return - (s_props * mask * advantage).sum() / mask.sum()


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.torch = torch.cuda if args.use_cuda else torch
        self.bsz = args.batch_size
        self.rnn_hsz = args.rnn_hsz
        self.max_len = args.max_len

        self.src_emb = nn.Embedding(
            args.src_vs, args.emb_dim, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(
            args.tgt_vs, args.emb_dim, padding_idx=PAD)
        self.enc = nn.LSTM(args.emb_dim, args.rnn_hsz, 1,
                           batch_first=True,
                           dropout=args.dropout)
        self.dec_hidden = nn.Linear(args.rnn_hsz, args.rnn_hsz)
        self.dec = nn.LSTM(args.rnn_hsz, args.rnn_hsz, 1,
                           batch_first=True,
                           dropout=args.dropout)
        self.out = nn.Linear(self.rnn_hsz, args.tgt_vs)

    def encode(self, src_inp):
        emb = self.src_emb(src_inp)
        _, (hidden, _) = self.enc(emb)
        dec_hidden = self.dec_hidden(hidden)

        weight = next(self.parameters()).data
        c = Variable(weight.new(1, self.bsz, self.rnn_hsz).zero_())

        return (dec_hidden.contiguous(), c.contiguous())

    def forward(self, src_inp, tgt_inp):
        hiiden = self.encode(src_inp)
        word = Variable(self.torch.LongTensor([[BOS]] * self.bsz))
        emb = self.tgt_emb(word)
        outputs = []

        for i in range(self.max_len):
            _, hiiden = self.dec(emb, hiiden)
            props = F.log_softmax(self.out(hiiden[0][-1]), dim=-1)
            emb = self.tgt_emb(tgt_inp[:, i]).unsqueeze(1)

            outputs.append(props.unsqueeze(1))

        return torch.cat(outputs, 1)

    def sample(self, src_inp, max_props=True):
        hiiden = self.encode(src_inp)
        word = Variable(self.torch.LongTensor([[BOS]] * self.bsz))
        emb = self.tgt_emb(word)
        outputs, words = [], []

        for i in range(self.max_len):
            _, hiiden = self.dec(emb, hiiden)
            props = F.log_softmax(self.out(hiiden[0][-1]), dim=-1)

            if max_props:
                _, word = props.max(-1, keepdim=True)
            else:
                _props = props.data.clone().exp()
                word = Variable(_props.multinomial(1))
                outputs.append(props.unsqueeze(1))

            emb = self.tgt_emb(word)
            words.append(word)

        if max_props:
            return torch.cat(words, 1)

        else:
            return torch.cat(words, 1), torch.cat(outputs, 1)
