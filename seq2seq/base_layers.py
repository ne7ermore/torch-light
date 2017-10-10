import torch
import torch.nn as nn
import numpy as np

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_hid))
        self.bias = nn.Parameter(torch.zeros(d_hid))

    def forward(self, z):
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1).clamp(min=self.eps)
        ln_out = (z - mu) / sigma
        return ln_out * self.weight.expand_as(ln_out) + self.bias.expand_as(ln_out)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, attn_dropout=0.1):
        super().__init__()
        self.temper = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, attn_mask=None):
        # (n_head*bsz) * len_q * len_k
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        if attn_mask is not None:
            assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask, -float('inf'))

        s1, s2, s3 = attn.size()
        attn = self.softmax(attn.view(s1*s2, s3)).view(s1, s2, s3)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
