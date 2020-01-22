import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.temperature = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(d_k)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.layer_norm(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = output + residual

        return output, attn


class PositionWise(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1),
            nn.ReLU(),
            nn.Conv1d(d_ff, d_model, 1),
            nn.Dropout(dropout)
        )
        self.lm = nn.LayerNorm(d_model)

    def forward(self, input):
        residual = input

        input = self.lm(input)

        out = self.seq(input.transpose(1, 2)).transpose(1, 2)
        return residual + out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_head, dropout=0.1):
        super().__init__()
        self.mh = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pw = PositionWise(d_model, d_ff, dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.mh(
            enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pw(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_head, dropout=0.1):
        super().__init__()
        self.slf_mh = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.dec_mh = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pw = PositionWise(d_model, d_ff, dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_mh(dec_input, dec_input,
                                               dec_input, slf_attn_mask)
        dec_output *= non_pad_mask
        m_dec_output = dec_output

        dec_output, dec_enc_attn = self.dec_mh(
            dec_output, enc_output, enc_output, dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pw(dec_output)
        dec_output *= non_pad_mask

        return dec_output, m_dec_output
