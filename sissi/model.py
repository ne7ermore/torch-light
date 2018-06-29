import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util import *


class Model(nn.Module):
    def __init__(self, dropout=0.5, time_hsz=300, note_size=100, layers=2):
        super().__init__()

        self.dropout = dropout
        self.time_hsz = time_hsz
        self.time_rnn = nn.LSTM(80,
                                hidden_size=time_hsz,
                                num_layers=2,
                                dropout=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(TIME_AXIS_UNITS)

        self.style_l.weight.data.uniform_(-stdv, stdv)
        self.style_l.bias.data.fill_(0)

    def forward(self, input_mat, output_mat):
        input_slice = input_mat[:, 0:-1]
        n_batch, n_time, n_note, n_ipn = input_slice.size()

        time_inputs = input_slice.transpose(
            1, 0).view(n_time, n_batch * n_note, n_ipn)
        time_result, _ = self.time_rnn(time_inputs)
        time_result = F.dropout(time_result, p=self.dropout)

        time_result = time_result.view(n_time, n_batch, n_note, self.time_hsz)
        time_result = time_result.transpose(0, 2)
        time_result = time_result.view(n_note, n_batch * n_time, self.time_hsz)
