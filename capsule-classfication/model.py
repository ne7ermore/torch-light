import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F

import const


def squash(input):
    mag_sq = torch.sum(input**2, dim=2, keepdim=True)
    mag = torch.sqrt(mag_sq)
    out = (mag_sq / (1.0 + mag_sq)) * (input / mag)
    return out


def to_one_hot(x, length, use_cuda, is_zero=True):
    bsz, x_list = x.size(0), x.data.tolist()
    x_one_hot = torch.zeros(bsz, length)
    if is_zero:
        for i in range(bsz):
            x_one_hot[i, x_list[i]] = 1.
    else:
        x_one_hot = x_one_hot + .1
        for i in range(bsz):
            x_one_hot[i, x_list[i]] = -1.

    x_one_hot = Variable(x_one_hot)
    if use_cuda:
        x_one_hot = x_one_hot.cuda()

    return x_one_hot


class BiRNN(nn.Module):
    def __init__(self, vsz, embed_dim, dropout, hsz, layers):
        super().__init__()

        self.lookup_table = nn.Embedding(vsz, embed_dim, padding_idx=const.PAD)
        self.lstm = nn.LSTM(embed_dim, hsz, layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)

        scope = 1. / math.sqrt(vsz)
        self.lookup_table.weight.data.uniform_(-scope, scope)

    def forward(self, input):
        encode = self.lookup_table(input)
        lstm_out, _ = self.lstm(encode)
        feats = lstm_out.mean(1)

        return lstm_out, feats


class ConvUnit(nn.Module):
    def __init__(self):
        super(ConvUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels=256,
                              out_channels=32,
                              kernel_size=5,
                              stride=1)

    def forward(self, x):
        return self.conv(x)


class PrimaryCap(nn.Module):
    def __init__(self, num_primary_units):
        super().__init__()

        self.num_primary_units = num_primary_units
        self.convUnits = nn.ModuleList(
            [ConvUnit() for _ in range(num_primary_units)])

    def forward(self, input):
        bsz = input.size(0)
        # num_primary_units * (b*32*6*6)
        units = [unit(input) for unit in self.convUnits]
        units = torch.stack(units, dim=1)  # b*num_primary_units*32*6*6
        units = units.view(bsz, self.num_primary_units, -1)

        return squash(units)  # b*num_primary_units*(32*6*6)


class DigitCap(nn.Module):
    def __init__(self,
                 use_cuda,
                 num_primary_units,
                 labels,
                 output_unit_size,
                 primary_unit_size,
                 iterations):
        super().__init__()

        self.labels = labels
        self.use_cuda = use_cuda
        self.primary_unit_size = primary_unit_size
        self.iterations = iterations

        self.W = nn.Parameter(torch.randn(1,
                                          primary_unit_size,
                                          labels,
                                          output_unit_size,
                                          num_primary_units))

    def forward(self, input):
        bsz = input.size(0)
        input_t = input.transpose(1, 2)  # b*f*num_primary_units
        # b*f*l*num_primary_units*1
        u = torch.stack([input_t] * self.labels, dim=2).unsqueeze(4)
        # b*f*l*output_unit_size*num_primary_units
        W = torch.cat([self.W] * bsz, dim=0)
        # b*f*l*output_unit_size*1
        u_hat = torch.matmul(W, u)

        b_ij = Variable(torch.zeros(1, self.primary_unit_size, self.labels, 1))
        if self.use_cuda:
            b_ij = b_ij.cuda()

        for _ in range(self.iterations):
            c_ij = F.softmax(b_ij, dim=-1)
            c_ij = torch.cat([c_ij] * bsz, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)  # b*1*l*output_unit_size*1

            v_j1 = torch.cat([v_j] * self.primary_unit_size, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(
                4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1

        return v_j.squeeze()  # b*l*output_unit_size


class Capsule(nn.Module):
    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.rnn = BiRNN(self.vsz,
                         self.embed_dim,
                         self.dropout,
                         self.hsz,
                         self.layers)
        self.fc = nn.Linear(self.hsz * 2, self.max_len)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=7),
            nn.ReLU(inplace=True)
        )
        self.pCap = PrimaryCap(self.num_primary_units)
        self.dCap = DigitCap(self.use_cuda,
                             self.num_primary_units,
                             self.labels,
                             self.output_unit_size,
                             self.primary_unit_size,
                             self.iterations)
        self.recon = nn.Sequential(
            nn.Linear(self.output_unit_size, self.hsz),
            nn.ReLU(inplace=True),
            nn.Linear(self.hsz, self.hsz * 2),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        scope = 1. / math.sqrt(self.vsz)

        for m in self.modules():
            if type(m) == nn.Linear:
                m.weight.data.uniform_(-scope, scope)

    def forward(self, input):
        lstm_out, lstm_feats = self.rnn(input)

        in_capsule = self.fc(lstm_out)
        in_capsule = in_capsule.unsqueeze(1)  # b*1*28*28

        conv1_out = self.conv1(in_capsule)
        pCap_out = self.pCap(conv1_out)
        dCap_out = self.dCap(pCap_out)

        return dCap_out, lstm_feats

    def loss(self, props, target, lstm_feats):
        zero_t = to_one_hot(target, self.labels, self.use_cuda)
        unzero_t = to_one_hot(target, self.labels, self.use_cuda, False)

        return self.margin_loss(props, zero_t) + self.reconstruction_loss(lstm_feats, props, unzero_t) * 0.05
        # return self.margin_loss(props, zero_t)

    def margin_loss(self, props, target):
        bsz = props.size(0)
        v_abs = torch.sqrt((props**2).sum(dim=2, keepdim=True))
        zero = Variable(torch.zeros(1))
        if self.use_cuda:
            zero = zero.cuda()

        m_plus, m_minus = .9, .1
        max_pos = torch.max(zero, m_plus - v_abs).view(bsz, -1)**2
        max_neg = torch.max(zero, v_abs - m_minus).view(bsz, -1)**2

        loss = target * max_pos + .5 * (1. - target) * max_neg

        return loss.mean()

    def reconstruction_loss(self, lstm_feats, props, target):
        bsz, target = props.size(0), target.unsqueeze(2)
        v = torch.sqrt((props**2).sum(dim=2, keepdim=True))
        zero = Variable(torch.zeros(1))
        if self.use_cuda:
            zero = zero.cuda()

        r = self.recon(props)  # b*l*(hsz*2)
        lstm_feats = lstm_feats.unsqueeze(1)  # b*1*(hsz*2)

        _temp = (r * target * v * lstm_feats).sum(2, keepdim=True)
        loss = torch.max(zero, 1. + _temp)

        return loss.mean()
