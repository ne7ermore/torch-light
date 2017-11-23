import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class CNN_Ranking(nn.Module):
    def __init__(self, args):
        super().__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.embedded_chars_left = nn.Embedding(self.src_vocab_size, self.embed_dim)
        self.embedded_chars_right = nn.Embedding(self.tgt_vocab_size, self.embed_dim)

        self.conv_left, self.conv_right = [], []
        for i, filter_size in enumerate(self.filter_sizes):
            conv_left_name = "conv_left_%d" % i
            conv_right_name = "conv_right_%d" % i
            self.__setattr__(conv_left_name,
                             nn.Conv2d(in_channels=1,
                             out_channels=self.num_filters,
                             kernel_size=(filter_size, self.embed_dim)))
            self.conv_left.append(self.__getattr__(conv_left_name))

            self.__setattr__(conv_right_name,
                             nn.Conv2d(in_channels=1,
                             out_channels=self.num_filters,
                             kernel_size=(filter_size, self.embed_dim)))
            self.conv_right.append(self.__getattr__(conv_right_name))

        ins = len(self.filter_sizes) * self.num_filters
        self.simi_weight = nn.Parameter(torch.zeros(ins, ins))

        self.out_lr = nn.Linear(2*ins+1, self.hidden_size)
        self.logistic = nn.Linear(self.hidden_size, 2)

        self._init_weights()

    def forward(self, input_left, input_right):
        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3

        enc_left = self.embedded_chars_left(input_left)
        enc_right = self.embedded_chars_right(input_right)

        enc_left = enc_left.unsqueeze(c_idx)
        enc_right = enc_right.unsqueeze(c_idx)

        enc_outs_left, enc_outs_right = [], []
        for index, (encoder_left, encoder_right) in enumerate(zip(self.conv_left, self.conv_right)):
            enc_left_ = F.relu(encoder_left(enc_left))
            enc_right_ = F.relu(encoder_right(enc_right))

            h_left = enc_left_.size()[h_idx]
            h_right = enc_right_.size()[h_idx]

            enc_left_ = F.max_pool2d(enc_left_, kernel_size=(h_left, 1))
            enc_right_ = F.max_pool2d(enc_right_, kernel_size=(h_right, 1))

            enc_left_ = enc_left_.squeeze(w_idx)
            enc_left_ = enc_left_.squeeze(h_idx)
            enc_right_ = enc_right_.squeeze(w_idx)
            enc_right_ = enc_right_.squeeze(h_idx)

            enc_outs_left.append(enc_left_)
            enc_outs_right.append(enc_right_)

        hid_in_left = torch.cat(enc_outs_left, c_idx)
        enc_outs_right = torch.cat(enc_outs_right, c_idx)

        transform_left = torch.mm(hid_in_left, self.simi_weight)
        sims = torch.sum(torch.mm(transform_left,
                enc_outs_right.t()), dim=c_idx, keepdim=True)

        new_input = torch.cat([hid_in_left, sims, enc_outs_right], c_idx)

        out = F.dropout(self.out_lr(new_input), p=self.dropout)
        return F.log_softmax(self.logistic(out))

    def _init_weights(self, scope=1.):
        self.embedded_chars_left.weight.data.uniform_(-scope, scope)
        self.embedded_chars_right.weight.data.uniform_(-scope, scope)
        init.xavier_uniform(self.simi_weight)
        init.xavier_uniform(self.out_lr.weight)
        init.xavier_uniform(self.logistic.weight)
