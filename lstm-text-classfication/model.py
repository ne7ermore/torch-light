import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F

import const


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)


class LSTM_Text(nn.Module):
    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.num_directions = 2 if self.bidirectional else 1

        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim,
                                         padding_idx=const.PAD)
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.lstm_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.ln = LayerNorm(self.hidden_size * self.num_directions)
        self.logistic = nn.Linear(self.hidden_size * self.num_directions,
                                  self.label_size)

        self._init_weights()

    def _init_weights(self, scope=1.):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        self.logistic.weight.data.uniform_(-scope, scope)
        self.logistic.bias.data.fill_(0)

    def init_hidden(self):
        num_layers = self.lstm_layers * self.num_directions

        weight = next(self.parameters()).data
        return (Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_()), Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_()))

    def forward(self, input, hidden):
        encode = self.lookup_table(input)
        lstm_out, hidden = self.lstm(encode.transpose(0, 1), hidden)
        output = self.ln(lstm_out)[-1]
        return F.log_softmax(self.logistic(output)), hidden
