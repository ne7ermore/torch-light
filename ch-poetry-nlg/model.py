import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

from const import *

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.num_directions = 2 if self.bidirectional else 1
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim,
                    self.hidden_size,
                    self.lstm_layers,
                    batch_first=True,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional)
        self.lr = nn.Linear(self.hidden_size*self.num_directions,
                        self.vocab_size)

        self._init_weights()

    def _init_weights(self, scope=.1):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        self.lr.weight.data.uniform_(-scope, scope)
        self.lr.bias.data.fill_(0)

    def init_hidden(self, bsz):
        num_layers = self.lstm_layers*self.num_directions

        weight = next(self.parameters()).data
        return (Variable(weight.new(num_layers, bsz, self.hidden_size).zero_()),
                Variable(weight.new(num_layers, bsz, self.hidden_size).zero_()))

    def forward(self, input, hidden):
        encode = self.lookup_table(input)
        lstm_out, hidden = self.lstm(encode, hidden)
        lstm_out = F.dropout(lstm_out, p=self.dropout)
        out = self.lr(lstm_out.contiguous().view(-1, lstm_out.size(2)))
        return F.log_softmax(out, dim=-1), hidden

