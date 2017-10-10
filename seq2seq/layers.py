import torch
import torch.nn as nn
import torch.nn.init as init

from base_layers import ScaledDotProductAttention, LayerNormalization

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_v = self.d_k = d_k = d_model // n_head

        for name in ["w_qs", "w_ks", "w_vs"]:
            self.__setattr__(name,
                nn.Parameter(torch.FloatTensor(n_head, d_model, d_k)))

        self.attention = ScaledDotProductAttention(d_k)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False) # d_k*n_head == d_model
        self.dropout = nn.Dropout(dropout)

        self._init_weight()

    def forward(self, q, k, v, attn_mask):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual = q

        bsz, len_q, d_model = q.size()
        _, len_k, _ = k.size()
        _, len_v, _ = v.size()

        def reshape(x):
            """[bsz, len, d_*] -> [n_head x (bsz*len) x d_*]"""
            return x.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        q_s, k_s, v_s = map(reshape, [q, k, v])

        # first linears for each of q, k, v
        # [n_head x (bsz*len_*) x d_model] -> [n_head*bsz x len_* x d_*]
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)

        # perform attention, result size = (n_head * bsz) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))

        # back to original bsz batch, result size = bsz x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, bsz, dim=0), dim=-1).view(-1, n_head*d_v)
        outputs = self.dropout(self.proj(outputs)).view(bsz, len_q, -1)
        return self.layer_norm(outputs + residual), attns

    def _init_weight(self):
        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)
        init.xavier_normal(self.proj.weight)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn
