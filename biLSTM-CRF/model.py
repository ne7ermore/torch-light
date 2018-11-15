import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init


class BiLSTM_CRF_Size(nn.Module):
    def __init__(self, args):
        super(BiLSTM_CRF_Size, self).__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.bi_lstm = nn.LSTM(self.embed_dim,
                               self.lstm_hsz,
                               num_layers=self.lstm_layers,
                               batch_first=True,
                               dropout=self.dropout,
                               bidirectional=True)

        self.logistic = nn.Linear(2 * self.lstm_hsz, self.tag_size)
        self._init_weights(scope=self.w_init)

    def forward(self, sentences):
        sents_ebd = self.lookup_table(sentences)
        output, _ = self.bi_lstm(sents_ebd)
        output = self.logistic(output).view(-1, self.tag_size)
        return F.log_softmax(output)

    def _init_weights(self, scope=0.25):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        init.xavier_uniform(self.logistic.weight)
