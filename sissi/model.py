import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util import *


class Model(nn.Module):
    def __init__(self,
                 use_cuda,
                 dropout=0.5,
                 time_hsz=300,
                 note_size=100,
                 layers=2):
        super().__init__()

        self.dropout = dropout
        self.time_hsz = time_hsz
        self.note_size = note_size
        self.layers = layers
        self.use_cuda = use_cuda

        self.time_rnn = nn.LSTM(80,
                                hidden_size=time_hsz,
                                num_layers=2,
                                dropout=dropout)

        self.note_rnn = nn.LSTM(self.time_hsz + 2,
                                hidden_size=note_size,
                                num_layers=2,
                                dropout=dropout)

        self.lr = nn.Linear(note_size, 2)

        self._reset_parameters()

    def _new_start_note(self, size):
        return next(self.parameters()).data.new(*size).zero_()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.time_hsz)

        self.lr.weight.data.uniform_(-stdv, stdv)
        self.lr.bias.data.fill_(0)

    def _init_time_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.layers, bsz, self.time_hsz).zero_(),
                weight.new(self.layers, bsz, self.time_hsz).zero_())

    def _init_note_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.layers, bsz, self.note_size).zero_(),
                weight.new(self.layers, bsz, self.note_size).zero_())

    def _gen_rand(self):
        x = torch.rand(1)[0]
        if self.use_cuda:
            return x.cuda()
        return x

    def _gen_inp(self, chosen, time):
        if self.use_cuda:
            chosen = chosen.cpu()
        inp_t = noteStateSingleToInputForm(chosen.numpy(), time)
        inp_t = torch.from_numpy(np.asarray(inp_t, dtype=np.float32))
        if self.use_cuda:
            inp_t = inp_t.cuda()
        return inp_t.unsqueeze(0)

    def forward(self, input_mat, output_mat):
        input_slice = input_mat[:, 0:-1]
        n_batch, n_time, n_note, n_ipn = input_slice.size()

        time_hidden = self._init_time_hidden(n_batch * n_note)
        note_hidden = self._init_note_hidden(n_batch * n_time)

        time_inputs = input_slice.transpose(
            1, 0).contiguous().view(n_time, n_batch * n_note, n_ipn)
        time_result, _ = self.time_rnn(time_inputs, time_hidden)
        time_result = F.dropout(time_result, p=self.dropout)

        time_result = time_result.view(n_time, n_batch, n_note, self.time_hsz)
        time_result = time_result.transpose(0, 2).contiguous()
        time_result = time_result.view(n_note, n_batch * n_time, self.time_hsz)

        correct_choices = output_mat[:, 1:, :-1]
        correct_choices = correct_choices.transpose(0, 1)
        correct_choices = correct_choices.transpose(0, 2).contiguous()
        correct_choices = correct_choices.view(
            n_note - 1, n_batch * n_time, 2)

        start_note_values = self._new_start_note((1, n_batch * n_time, 2))
        note_choices_inputs = torch.cat((start_note_values, correct_choices))
        note_inputs = torch.cat((time_result, note_choices_inputs), dim=-1)

        note_result, _ = self.note_rnn(note_inputs, note_hidden)
        note_result = F.dropout(note_result, p=self.dropout)
        note_result = F.sigmoid(self.lr(note_result))

        note_final = note_result.view(n_note, n_batch, n_time, 2)
        note_final = note_final.transpose(0, 1)
        note_final = note_final.transpose(1, 2).contiguous()

        return note_final

    def predict(self, inp, time_steps):
        n_note = inp.size()[0]

        time_hidden = self._init_time_hidden(n_note)
        note_hidden = self._init_note_hidden(1)

        inp_t = inp.unsqueeze(0)  # 1*n_note*n_ipn
        the_choice = self._new_start_note((n_note, 1, 2))
        out = []

        for time in range(time_steps):
            time_result, time_hidden = self.time_rnn(
                inp_t, time_hidden)  # 1*n_note*time_hsz
            time_result = time_result.transpose(0, 1).contiguous()

            note_inputs = torch.cat((time_result, the_choice), dim=-1)
            note_result, note_hidden = self.note_rnn(note_inputs, note_hidden)
            note_result = F.sigmoid(self.lr(note_result))

            shouldPlay = self._gen_rand() < note_result[:, 0, 0]
            shouldArtic = shouldPlay * \
                (self._gen_rand() < note_result[:, 0, 1])

            chosen = torch.stack((shouldPlay, shouldArtic),
                                 dim=-1).to(torch.float32)
            the_choice = chosen.unsqueeze(1)
            inp_t = self._gen_inp(chosen, time)
            out.append(chosen)

        return torch.stack(out)


class MNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_mat, note_final):
        active_notes = (output_mat[:, 1:, :, 0]).unsqueeze(3)
        mask = torch.cat(
            (torch.ones_like(active_notes.data), active_notes), dim=-1)

        llh = mask * \
            torch.log(
                (2 * note_final * output_mat[:, 1:] - note_final - output_mat[:, 1:] + 1.).clamp(min=1e-12))
        loss = -llh.sum() / mask.sum()

        return loss
