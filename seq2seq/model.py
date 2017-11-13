import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from layers import EncoderLayer, DecoderLayer
from utils import *
from const import *

class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, max_word_len, n_enc, d_model, d_ff, n_head, dropout):
        super().__init__()

        self.n_position = max_word_len + 1
        self.enc_vocab_size = enc_vocab_size
        self.d_model = d_model

        self.enc_ebd = nn.Embedding(enc_vocab_size,
                            d_model, padding_idx=PAD)
        self.pos_ebd = nn.Embedding(self.n_position,
                            d_model, padding_idx=PAD)
        self.encodes = nn.ModuleList([
            EncoderLayer(d_model, d_ff, n_head, dropout) for _ in range(n_enc)])

        self._init_weight()

    def _init_weight(self, scope=.1):
        self.enc_ebd.weight.data.uniform_(-scope, scope)
        self.pos_ebd.weight.data = position(self.n_position, self.d_model)
        self.pos_ebd.weight.requires_grad = False

    def forward(self, input, pos):
        encode = self.enc_ebd(input) + self.pos_ebd(pos)
        enc_outputs, enc_output = [], encode
        slf_attn_mask = get_attn_padding_mask(input, input)

        for layer in self.encodes:
            enc_output = layer(enc_output, slf_attn_mask)
            enc_outputs.append(enc_output)

        return enc_outputs

class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_word_len, n_dec, d_model, d_ff, n_head, dropout):
        super().__init__()

        self.d_model = d_model
        self.n_position = max_word_len + 1

        self.pos_ebd = nn.Embedding(
            self.n_position, d_model, padding_idx=PAD)
        self.dec_ebd = nn.Embedding(
            dec_vocab_size, d_model, padding_idx=PAD)
        self.decodes = nn.ModuleList([
            DecoderLayer(d_model, d_ff, n_head, dropout) for _ in range(n_dec)])

        self._init_weight()

    def _init_weight(self, scope=.1):
        self.dec_ebd.weight.data.uniform_(-scope, scope)
        self.pos_ebd.weight.data = position(self.n_position, self.d_model)
        self.pos_ebd.weight.requires_grad = False

    def forward(self, enc_outputs, enc_input, dec_input, dec_pos):
        dec_output = self.dec_ebd(dec_input) + self.pos_ebd(dec_pos)

        dec_slf_attn_mask = torch.gt(
            get_attn_padding_mask(dec_input, dec_input) + get_attn_subsequent_mask(dec_input), 0)
        dec_enc_attn_pad_mask = get_attn_padding_mask(dec_input, enc_input)

        for layer, enc_output in zip(self.decodes, enc_outputs):
            dec_output = layer(dec_output, enc_output,
                dec_slf_attn_mask, dec_enc_attn_pad_mask)

        return dec_output

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.enc = Encoder(self.enc_vocab_size, self.max_word_len, self.n_stack_layers, self.d_model, self.d_ff, self.n_head, self.dropout)
        self.dec = Decoder(self.dec_vocab_size, self.max_word_len, self.n_stack_layers,
                self.d_model, self.d_ff, self.n_head, self.dropout)

        self.linear = nn.Linear(self.d_model, self.dec_vocab_size,  bias=False)

        self._init_weight()

    def _init_weight(self):
        if self.share_linear:
            self.linear.weight = self.dec.dec_ebd.weight
        else:
            init.xavier_normal(self.linear.weight)

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())

    def forward(self, src, src_pos, tgt, tgt_pos):
        tgt, tgt_pos = tgt[:, :-1], tgt_pos[:, :-1]

        enc_outputs = self.enc(src, src_pos)
        dec_output = self.dec(enc_outputs, src, tgt, tgt_pos)

        out = self.linear(dec_output)

        return F.log_softmax(out.view(-1, self.dec_vocab_size))
