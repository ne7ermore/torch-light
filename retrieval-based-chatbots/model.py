import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.emb = nn.Embedding(self.dict_size, self.emb_dim)
        self.first_gru = nn.GRU(input_size=self.emb_dim,
                                hidden_size=self.first_rnn_hsz,
                                num_layers=1,
                                batch_first=True)
        self.transform_A = nn.Linear(
            self.first_rnn_hsz, self.first_rnn_hsz, bias=False)
        self.cnn = nn.Conv2d(in_channels=2,
                             out_channels=self.fillters,
                             kernel_size=self.kernel_size)
        self.match_vec = nn.Linear(16 * 16 * 8, self.match_vec_dim)
        self.second_gru = nn.GRU(input_size=self.match_vec_dim,
                                 hidden_size=self.second_rnn_hsz,
                                 num_layers=1)
        self.pred = nn.Linear(self.match_vec_dim, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.emb_dim)

        self.emb.weight.data.uniform_(-stdv, stdv)

        self.transform_A.weight.data.uniform_(-stdv, stdv)

        self.match_vec.weight.data.uniform_(-stdv, stdv)
        self.match_vec.bias.data.fill_(0)

        self.pred.weight.data.uniform_(-stdv, stdv)
        self.pred.bias.data.fill_(0)

    def forward(self, utterances, responses):
        bsz = utterances.size(0)

        resps_emb = self.emb(responses)
        resps_gru, _ = self.first_gru(resps_emb)
        resps_gru = F.dropout(resps_gru, p=self.dropout)

        resps_emb_t = resps_emb.transpose(1, 2)
        resps_gru_t = resps_gru.transpose(1, 2)
        uttes_t = utterances.transpose(0, 1)

        match_vecs = []
        for utte in uttes_t:
            utte_emb = self.emb(utte)
            mat_1 = torch.matmul(utte_emb, resps_emb_t)
            utte_gru, _ = self.first_gru(utte_emb)
            utte_gru = F.dropout(utte_gru, p=self.dropout)
            mat_2 = torch.matmul(self.transform_A(utte_gru), resps_gru_t)

            M = torch.stack([mat_1, mat_2], 1)
            cnn_layer = F.relu(self.cnn(M))
            pool_layer = F.max_pool2d(cnn_layer,
                                      self.kernel_size,
                                      stride=self.kernel_size)
            pool_layer = pool_layer.view(bsz, -1)
            match_vec = F.relu(self.match_vec(pool_layer))
            match_vecs.append(match_vec)

        match_vecs = torch.stack(match_vecs, 0)
        match_vecs = F.dropout(match_vecs, p=self.dropout)
        _, hidden = self.second_gru(match_vecs)
        hidden = F.dropout(hidden[-1], p=self.dropout)
        props = F.log_softmax(self.pred(hidden), dim=-1)

        return props
