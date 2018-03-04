import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision.models import inception_v3

import numpy as np

from const import BOS, PAD

class Attention(nn.Module):
    def __init__(self, hsz):
        super().__init__()

        self.hsz = hsz

        self.sigma = nn.Linear(hsz, hsz)
        self.beta = nn.Linear(hsz, hsz, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.hsz)

        self.sigma.weight.data.uniform_(-stdv, stdv)
        self.beta.weight.data.uniform_(-stdv, stdv)

    def forward(self, hiddens, hidden):
        hiddens.append(hidden)
        sigma = torch.tanh(self.sigma(hidden))
        _hiddens = torch.stack(hiddens, dim=1)
        _betas = torch.sum(torch.exp(self.beta(_hiddens)), dim=1)
        beta = torch.exp(self.beta(sigma)) / _betas

        return (beta*hidden).unsqueeze(1)

class Actor(nn.Module):
    def __init__(self, vocab_size, dec_hsz, rnn_layers, bsz, max_len, dropout, use_cuda):
        super().__init__()

        self.torch = torch.cuda if use_cuda else torch
        self.dec_hsz = dec_hsz
        self.rnn_layers = rnn_layers
        self.bsz = bsz
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.enc = inception_v3(True)
        self.enc_out = nn.Linear(1000, dec_hsz)
        self.lookup_table = nn.Embedding(vocab_size, dec_hsz, padding_idx=PAD)
        self.rnn = nn.LSTM(dec_hsz+dec_hsz, dec_hsz, rnn_layers,
                           batch_first=True,
                           dropout=dropout)
        self.attn = Attention(dec_hsz)
        self.out = nn.Linear(self.dec_hsz, vocab_size)

        self._reset_parameters()

    def encode(self, imgs):
        if self.training:
            enc = self.enc(imgs)[0]
        else:
            enc = self.enc(imgs)
        enc = self.enc_out(enc)
        return enc

    def feed_enc(self, enc):
        weight = next(self.parameters()).data
        c = Variable(weight.new(
                self.rnn_layers, self.bsz, self.dec_hsz).zero_())

        h = Variable(enc.data.
                unsqueeze(0).expand(self.rnn_layers, *enc.size()))

        return (h.contiguous(), c.contiguous())

    def forward(self, hidden, labels):
        word = Variable(self.torch.LongTensor([[BOS]]*self.bsz))
        emb_enc = self.lookup_table(word)
        hiddens = [hidden[0].squeeze()]
        attn = torch.transpose(hidden[0], 0, 1)
        outputs, words = [], [word]

        for i in range(self.max_len):
            _, hidden = self.rnn(torch.cat([emb_enc, attn], -1), hidden)
            props = self.out(hidden[0][-1])
            attn = self.attn(hiddens, hidden[0][-1])

            emb_enc = self.lookup_table(labels[:, i])
            emb_enc = emb_enc.unsqueeze(1)
            _, word = torch.max(props, -1, keepdim=True)

            outputs.append(props.unsqueeze(1))
            words.append(word)

        return F.log_softmax(torch.cat(outputs, 1), dim=-1), torch.cat(words, 1)

    def speak(self, hidden):
        word = Variable(self.torch.LongTensor([[BOS]]*self.bsz))
        emb_enc = self.lookup_table(word)
        hiddens = [hidden[0].squeeze()]
        attn = torch.transpose(hidden[0], 0, 1)
        outputs, words = [], []

        for _ in range(self.max_len):
            _, hidden = self.rnn(torch.cat([emb_enc, attn], -1), hidden)
            props = self.out(hidden[0][-1])
            attn = self.attn(hiddens, hidden[0][-1])

            _, word = torch.max(props, -1, keepdim=True)
            emb_enc = self.lookup_table(word)

            words.append(word)
            outputs.append(props.unsqueeze(1))

        return F.log_softmax(torch.cat(outputs, 1), dim=-1), torch.cat(words, 1)

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.vocab_size)

        self.enc_out.weight.data.uniform_(-stdv, stdv)
        self.lookup_table.weight.data.uniform_(-stdv, stdv)
        self.out.weight.data.uniform_(-stdv, stdv)

        for p in self.enc.parameters():
            p.requires_grad = False

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())

class Critic(nn.Module):
    def __init__(self, vocab_size, dec_hsz, rnn_layers, bsz, max_len, dropout, use_cuda):
        super().__init__()

        self.use_cuda = use_cuda
        self.dec_hsz = dec_hsz
        self.rnn_layers = rnn_layers
        self.bsz = bsz
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.lookup_table = nn.Embedding(vocab_size, dec_hsz, padding_idx=PAD)
        self.rnn = nn.LSTM(self.dec_hsz,
                    self.dec_hsz,
                    self.rnn_layers,
                    batch_first=True,
                    dropout=dropout)
        self.out = nn.Linear(self.dec_hsz, vocab_size)

        self._reset_parameters()

    def feed_enc(self, enc):
        weight = next(self.parameters()).data
        c = Variable(weight.new(
                self.rnn_layers, self.bsz, self.dec_hsz).zero_())

        h = Variable(enc.data.
                unsqueeze(0).expand(self.rnn_layers, *enc.size()))
        return (h.contiguous(), c.contiguous())

    def forward(self, inputs, hidden):
        emb_enc = self.lookup_table(inputs[:, :-1])
        rnn_out, _ = self.rnn(emb_enc, hidden)
        rnn_out = rnn_out.contiguous()
        props = self.out(rnn_out)
        _, words = torch.max(props, -1)

        return F.log_softmax(props, dim=-1), words

    def td_error(self, mask_rewards_A, mask_rewards_C, criterion_C):
        return criterion_C(mask_rewards_C, mask_rewards_A)

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.vocab_size)

        self.lookup_table.weight.data.uniform_(-stdv, stdv)
        self.out.weight.data.uniform_(-stdv, stdv)
