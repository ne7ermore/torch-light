# borrowed from https://github.com/jadore801120/attention-is-all-you-need-pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F

import numpy as np

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

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout):
        super().__init__()
        self.temper = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, attn_mask):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn.view(-1, attn.size(2))).view(*attn.size())
        attn = self.dropout(attn)
        return torch.bmm(attn, v)

class MultiHeadAtt(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_v = self.d_k = d_k = d_model // n_head

        for name in ["w_qs", "w_ks", "w_vs"]:
            self.__setattr__(name,
                nn.Parameter(torch.FloatTensor(n_head, d_model, d_k)))

        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.lm = LayerNorm(d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        self._init_weight()

    def forward(self, q, k, v, attn_mask):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual = q

        bsz, len_q, d_model = q.size()
        len_k, len_v = k.size(1), v.size(1)

        def reshape(x):
            """[bsz, len, d_*] -> [n_head x (bsz*len) x d_*]"""
            return x.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        q_s, k_s, v_s = map(reshape, [q, k, v])

        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)

        outputs = self.attention(q_s, k_s, v_s, attn_mask.repeat(n_head, 1, 1))
        outputs = torch.cat(torch.split(outputs, bsz, dim=0), dim=-1).view(-1, n_head*d_v)
        outputs = F.dropout(self.w_o(outputs), p=self.dropout).view(bsz, len_q, -1)
        return self.lm(outputs + residual)

    def _init_weight(self):
        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)
        init.xavier_normal(self.w_o.weight)

class PositionWise(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.seq = nn.Sequential(
                nn.Conv1d(d_model, d_ff, 1),
                nn.ReLU(),
                nn.Conv1d(d_ff, d_model, 1),
                nn.Dropout(dropout)
            )
        self.lm = LayerNorm(d_model)

    def forward(self, input):
        residual = input
        out = self.seq(input.transpose(1, 2)).transpose(1, 2)
        return self.lm(residual + out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super().__init__()
        self.mh = MultiHeadAtt(n_head, d_model, dropout)
        self.pw = PositionWise(d_model, d_ff, dropout)

    def forward(self, enc_input, slf_attn_mask):
        enc_output = self.mh(
            enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output = self.pw(enc_output)
        return enc_output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout=0.1):
        super().__init__()
        self.slf_mh = MultiHeadAtt(n_head, d_model, dropout)
        self.dec_mh = MultiHeadAtt(n_head, d_model, dropout)
        self.pw = PositionWise(d_model, d_ff, dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask, dec_enc_attn_mask):
        dec_output = self.slf_mh(dec_input, dec_input, dec_input, slf_attn_mask)
        dec_output = self.dec_mh(dec_output, enc_output, enc_output, dec_enc_attn_mask)
        dec_output = self.pw(dec_output)

        return dec_output
