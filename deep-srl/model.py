import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from const import *


class RnnDropout(nn.Module):
    def __init__(self, dropout_prob, hidden_size, is_cuda):
        super().__init__()

        self.mask = torch.bernoulli(torch.Tensor(
            1, hidden_size).fill_(1. - dropout_prob))
        if is_cuda:
            self.mask = self.mask.cuda()
        self.dropout_prob = dropout_prob

    def forward(self, input):
        input = input * self.mask
        input *= 1. / (1. - self.dropout_prob)

        return input


class HwLSTMCell(nn.Module):
    def __init__(self, isz, hsz, dropout_prob, is_cuda):
        super().__init__()

        self.hsz = hsz

        self.w_ih = nn.Parameter(torch.Tensor(6 * hsz, isz))
        self.w_hh = nn.Parameter(torch.Tensor(5 * hsz, hsz))
        self.b_ih = nn.Parameter(torch.Tensor(6 * hsz))

        self.rdropout = RnnDropout(dropout_prob, hsz, is_cuda)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hsz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = input.new_zeros(input.size(0), self.hsz)
            hidden = (hidden, hidden)

        hx, cx = hidden

        input = F.linear(input, self.w_ih, self.b_ih)
        gates = F.linear(hx, self.w_hh) + input[..., :-self.hsz]

        in_gate, forget_gate, cell_gate, out_gate, r_gate = gates.chunk(5, 1)
        in_gate, forget_gate, out_gate, r_gate = map(
            torch.sigmoid, [in_gate, forget_gate, out_gate, r_gate])
        cell_gate = torch.tanh(cell_gate)
        k = input[..., -self.hsz:]

        cy = forget_gate * cx + in_gate * cell_gate
        hy = r_gate * out_gate * F.tanh(cy) + (1. - r_gate) * k

        if self.training:
            hy = self.rdropout(hy)

        return hy, cy


class HwLSTMlayer(nn.Module):
    def __init__(self, isz, hsz, dropout_prob, is_cuda):
        super().__init__()

        self.cell = HwLSTMCell(isz, hsz, dropout_prob, is_cuda)

    def forward(self, input, reverse=True):
        output, hidden = [], None
        for i in range(len(input)):
            hidden = self.cell(input[i], hidden)
            output.append(hidden[0])

        if reverse:
            output.reverse()

        return torch.stack(output)


class DeepBiLSTMModel(nn.Module):
    def __init__(self, vsz, lsz, ebd_dim, lstm_hsz, lstm_layers, dropout_prob, is_cuda, ebd_weights=None):
        super().__init__()

        self.ebd_weights = ebd_weights
        self.ebd = nn.Embedding(vsz, ebd_dim, padding_idx=PAD)
        self.lstms = nn.ModuleList([HwLSTMlayer(lstm_hsz, lstm_hsz, dropout_prob, is_cuda) if layer != 0 else HwLSTMlayer(
            ebd_dim, lstm_hsz, dropout_prob, is_cuda) for layer in range(lstm_layers)])
        self.logistic = nn.Linear(lstm_hsz, lsz)

        self.reset_parameters(ebd_dim)

    def reset_parameters(self, ebd_dim):
        stdv = 1.0 / math.sqrt(ebd_dim)
        self.logistic.weight.data.uniform_(-stdv, stdv)
        if self.ebd_weights is None:
            self.ebd.weight.data.uniform_(-stdv, stdv)
        else:
            self.ebd.weight.data.copy_(torch.from_numpy(self.ebd_weights))

    def forward(self, inp):
        inp = self.ebd(inp)
        inp = inp.permute(1, 0, 2)

        for rnn in self.lstms:
            inp = rnn(inp)

        inp = inp.permute(1, 0, 2).contiguous().view(-1, inp.size(2))
        out = self.logistic(inp)
        return F.log_softmax(out, dim=-1)


if __name__ == "__main__":
    d = DeepBiLSTMModel(30, 3, 300, 100, 8, 0.1, False)
    input = torch.LongTensor([[0, 1, 2], [2, 1, 2]])
    out = d(input)
    print(out.shape)
