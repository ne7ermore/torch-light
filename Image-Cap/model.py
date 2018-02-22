import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision.models import inception_v3

import numpy as np

from const import BOS, PAD

def Encode(use_cuda):
    enc = inception_v3(True)
    if use_cuda: enc = enc.cuda()
    return enc

class Actor(nn.Module):
    def __init__(self, vocab_size, dec_hsz, rnn_layers, bsz, max_len, dropout, use_cuda):
        super().__init__()

        self.torch = torch.cuda if use_cuda else torch
        self.dec_hsz = dec_hsz
        self.rnn_layers = rnn_layers
        self.bsz = bsz
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.enc_out = nn.Linear(1000, dec_hsz)
        self.lookup_table = nn.Embedding(vocab_size, dec_hsz, padding_idx=PAD)
        self.rnn = nn.LSTM(self.dec_hsz,
                    self.dec_hsz,
                    self.rnn_layers,
                    batch_first=True,
                    dropout=dropout)
        self.out = nn.Linear(self.dec_hsz, self.dec_hsz, bias=False)

        self._reset_parameters()

    def feed_enc(self, enc):
        weight = next(self.parameters()).data
        c = Variable(weight.new(
                self.rnn_layers, self.bsz, self.dec_hsz).zero_())

        enc = self.enc_out(enc)
        h = Variable(enc.data.
                unsqueeze(0).expand(self.rnn_layers, *enc.size()))
        return (h.contiguous(), c.contiguous())

    def forward(self, hidden):
        x = Variable(self.torch.LongTensor([[BOS]]*self.bsz))
        outputs = []

        for _ in range(0, self.max_len):
            emb_enc = self.lookup_table(x)
            _, hidden = self.rnn(emb_enc, hidden)
            props = F.linear(self.out(hidden[0][-1]), self.lookup_table.weight)
            _, next_idx = torch.max(props, -1, keepdim=True)
            x = torch.cat((x, next_idx), -1)
            outputs.append(props.unsqueeze(1))

        return F.log_softmax(torch.cat(outputs, 1), dim=-1), x

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.vocab_size)

        self.enc_out.weight.data.uniform_(-stdv, stdv)
        self.lookup_table.weight.data.uniform_(-stdv, stdv)
        self.out.weight.data.uniform_(-stdv, stdv)

class Critic(nn.Module):
    def __init__(self, vocab_size, dec_hsz, rnn_layers, bsz, max_len, dropout, use_cuda):
        super().__init__()

        self.use_cuda = use_cuda
        self.dec_hsz = dec_hsz
        self.rnn_layers = rnn_layers
        self.bsz = bsz
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.enc_out = nn.Linear(1000, dec_hsz)
        self.lookup_table = nn.Embedding(vocab_size, dec_hsz, padding_idx=PAD)
        self.rnn = nn.LSTM(self.dec_hsz,
                    self.dec_hsz,
                    self.rnn_layers,
                    batch_first=True,
                    dropout=dropout)
        self.out = nn.Linear(self.dec_hsz, self.dec_hsz, bias=False)

        self._reset_parameters()

    def feed_enc(self, enc):
        weight = next(self.parameters()).data
        c = Variable(weight.new(
                self.rnn_layers, self.bsz, self.dec_hsz).zero_())

        enc = self.enc_out(enc)
        h = Variable(enc.data.
                unsqueeze(0).expand(self.rnn_layers, *enc.size()))
        return (h.contiguous(), c.contiguous())

    def forward(self, inputs, hidden):
        emb_enc = self.lookup_table(inputs[:, :-1])
        rnn_out, _ = self.rnn(emb_enc, hidden)
        rnn_out = rnn_out.contiguous()
        props = F.linear(self.out(rnn_out), self.lookup_table.weight)
        _, words = torch.max(props, -1)

        return F.log_softmax(props, dim=-1), words

    def td_error(self, scores_A, scores_C, props_A, props_C, criterion_C):
        scores_A = scores_A.view(self.bsz, 1, 1)
        scores_C = scores_C.view(self.bsz, 1, 1)

        props_A = props_A.add(scores_A)
        props_C = props_C.add(scores_C)

        return criterion_C(props_C, props_A)

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.vocab_size)

        self.enc_out.weight.data.uniform_(-stdv, stdv)
        self.lookup_table.weight.data.uniform_(-stdv, stdv)
        self.out.weight.data.uniform_(-stdv, stdv)
