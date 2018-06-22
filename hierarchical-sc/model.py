import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from const import BOS, PAD


def multi_view_att(ori_memory, att_w, dec_hidden, *args):
    bsz, max_len, rnn_hsz = args

    dec_hidden = att_w(dec_hidden.squeeze())
    ori_memory_t = ori_memory.transpose(1, 2)

    beta_is = torch.exp(torch.tanh(torch.matmul(dec_hidden, ori_memory_t)))

    beta_i_sum = torch.sum(beta_is, 0, keepdim=True)
    beta_is = torch.div(beta_is, beta_i_sum)

    return torch.sum(torch.matmul(beta_is, ori_memory), dim=0)


class Model(nn.Module):
    def __init__(self, args, use_cuda):

        super().__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.torch = torch.cuda if use_cuda else torch

        self.emb = nn.Embedding(self.dict_size, self.emb_dim)
        self.encode = torch.nn.LSTM(input_size=self.emb_dim,
                                    hidden_size=self.rnn_hsz,
                                    num_layers=1,
                                    bidirectional=True)
        self.decode = torch.nn.LSTM(input_size=self.rnn_hsz,
                                    hidden_size=self.rnn_hsz,
                                    num_layers=1)
        self.summ_att_w = nn.Linear(self.rnn_hsz,
                                    self.rnn_hsz,
                                    bias=False)
        self.cls_att_w = nn.Linear(self.rnn_hsz,
                                   self.rnn_hsz,
                                   bias=False)
        self.summ_gen = nn.Linear(self.rnn_hsz, self.dict_size)
        self.cls_pred = nn.Linear(
            (self.max_ori_len + self.max_sum_len) * self.rnn_hsz, self.label_size)

        self.dropout = nn.Dropout(self.dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.emb_dim)

        self.emb.weight.data.uniform_(-stdv, stdv)

        self.summ_att_w.weight.data.uniform_(-stdv, stdv)

        self.cls_att_w.weight.data.uniform_(-stdv, stdv)

        self.summ_gen.weight.data.uniform_(-stdv, stdv)
        self.summ_gen.bias.data.fill_(0)

        self.cls_pred.weight.data.uniform_(-stdv, stdv)
        self.cls_pred.bias.data.fill_(0)

    def forward(self, original):
        bsz = original.size(0)

        ori_emb = self.emb(original)
        ori_emb_t = ori_emb.transpose(0, 1)

        encodes, (h, _) = self.encode(ori_emb_t)
        ori_memory = encodes[:, :, :self.rnn_hsz] + \
            encodes[:, :, self.rnn_hsz:]

        ori_hidden = (h[0] + h[1]).unsqueeze(0)
        c = Variable(next(self.parameters()).data.new(
            1, bsz, self.rnn_hsz).zero_())

        ori_memory = self.dropout(ori_memory)
        ori_hidden = self.dropout(ori_hidden)

        dec_hidden = (ori_hidden, c)

        word = Variable(self.torch.LongTensor([[BOS]] * bsz))
        v_ts, summ_props = [], []
        for _ in range(self.max_sum_len):
            summ_emb = self.emb(word).transpose(0, 1)
            _, dec_hidden = self.decode(summ_emb, dec_hidden)
            h_state = self.dropout(dec_hidden[0])

            v_c = multi_view_att(ori_memory,
                                 self.summ_att_w,
                                 h_state, bsz,
                                 self.max_ori_len,
                                 self.rnn_hsz)
            v_t = multi_view_att(ori_memory,
                                 self.cls_att_w,
                                 h_state, bsz,
                                 self.max_ori_len,
                                 self.rnn_hsz)

            props = F.log_softmax(self.summ_gen(v_c), -1)
            _, word = torch.max(props, -1, keepdim=True)

            v_ts.append(v_t.unsqueeze(1))
            summ_props.append(props.unsqueeze(1))

        summ_props = torch.cat(summ_props, 1)
        v_ts = self.dropout(torch.cat(v_ts, 1))
        r = torch.cat([v_ts, ori_memory.transpose(0, 1)], 1)
        l_props = self.cls_pred(r.view(bsz, -1))

        return summ_props, l_props


class NlpCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, props, tgt):
        tgt_props = props.gather(2, tgt.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        return -(tgt_props * mask).sum() / mask.sum()


class ScheduledOptim(object):
    def __init__(self, optimizer, parameters, lr, clip):
        self.optimizer = optimizer
        self.parameters = parameters
        self.n_current_epochs = 0
        self.lr = lr
        self.clip = clip

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.parameters, self.clip)

    def update_learning_rate(self):
        self.n_current_epochs += 1

        if self.n_current_epochs > 4:
            self.lr = self.lr / 2
            print("| learning rate updated - {}".format(self.lr))
            print('-' * 90)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
