import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from layers import EncoderLayer, DecoderLayer
import common
from const import *
import const


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, props, tgt):
        tgt_props = props.gather(2, tgt.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        return -(tgt_props * mask).sum() / (mask.sum() + 1e-9)


class Encoder(nn.Module):
    def __init__(self, n_enc, d_model, d_ff, d_k, d_v, n_head, dropout):
        super().__init__()

        self.encodes = nn.ModuleList([
            EncoderLayer(d_model, d_ff, d_k, d_v, n_head, dropout) for _ in range(n_enc)])

    def forward(self, encode, slf_attn_mask, non_pad_mask):
        enc_output = encode
        for layer in self.encodes:
            enc_output, _ = layer(enc_output, non_pad_mask, slf_attn_mask)
        return enc_output


class Decoder(nn.Module):
    def __init__(self, n_dec, d_model, d_ff, d_k, d_v, n_head, dropout):
        super().__init__()

        self.decodes = nn.ModuleList([
            DecoderLayer(d_model, d_ff, d_k, d_v, n_head, dropout) for _ in range(n_dec)])

    def forward(self, dec_output, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask):
        for layer in self.decodes:
            dec_output, m_dec_output = layer(dec_output, enc_output,
                                             non_pad_mask, slf_attn_mask, dec_enc_attn_mask)

        return dec_output, m_dec_output


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.n_position = args.max_context_len
        self.turn_embedding = nn.Embedding(
            args.turn_size, args.d_model, padding_idx=const.PAD)
        self.word_embedding = nn.Embedding(
            args.vocab_size, args.d_model, padding_idx=const.PAD)
        self.pos_embedding = nn.Embedding.from_pretrained(
            common.get_sinusoid_encoding_table(self.n_position, args.d_model, padding_idx=const.PAD))

        self.enc = Encoder(args.n_stack_layers, args.d_model,
                           args.d_ff, args.d_k, args.d_v, args.n_head, args.dropout)
        self.dec = Decoder(args.n_stack_layers, args.d_model,
                           args.d_ff, args.d_k, args.d_v, args.n_head, args.dropout)

        self.encode_w = nn.Linear(args.d_model, args.d_model, bias=False)
        self.decode_w = nn.Linear(args.d_model, args.d_model, bias=False)
        self.vt = nn.Linear(args.d_model, 1, bias=False)

        self.droupout = nn.Dropout(args.dropout)

        self._reset_parameters()

    def _reset_parameters(self, scope=.1):
        self.word_embedding.weight.data.uniform_(-scope, scope)
        self.turn_embedding.weight.data.uniform_(-scope, scope)

        for layer in self.modules():
            if type(layer) == nn.Linear:
                layer.weight.data.normal_(std=const.INIT_RANGE)

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())

    def forward(self, src_tensor, src_postion, turns_tensor, tgt_tensor):
        # encode embedding
        encode = self.word_embedding(
            src_tensor) + self.pos_embedding(src_postion) + self.turn_embedding(turns_tensor)
        encode = self.droupout(encode)

        # encode mask
        slf_attn_mask = common.get_attn_key_pad_mask(src_tensor, src_tensor)
        non_pad_mask = common.get_non_pad_mask(src_tensor)

        # encode
        enc_output = self.enc(encode, slf_attn_mask, non_pad_mask)

        # decode embedding
        dec_output = self.word_embedding(tgt_tensor)
        dec_output = self.droupout(dec_output)

        # decode mask
        non_pad_mask = common.get_non_pad_mask(tgt_tensor)
        slf_attn_mask_subseq = common.get_subsequent_mask(tgt_tensor)
        slf_attn_mask_keypad = common.get_attn_key_pad_mask(
            tgt_tensor, tgt_tensor, True)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = common.get_attn_key_pad_mask(
            src_tensor, tgt_tensor)

        # decode
        dec_output, m_dec_output = self.dec(
            dec_output, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)

        # pointer network
        distributes = self.attention(m_dec_output, enc_output)

        return distributes

    def attention(self, dec_output, last_enc_output):
        distributes = []
        last_enc_output = self.encode_w(last_enc_output)
        for step in range(dec_output.shape[1]):
            dec_slice = self.decode_w(dec_output[:, step]).unsqueeze(1)
            attn_encode = torch.tanh(dec_slice + last_enc_output)
            attn_encode = self.vt(attn_encode).squeeze(2)
            distributes.append(F.log_softmax(attn_encode, dim=-1) + 1e-9)

        return torch.stack(distributes, dim=1)

    def encode(self, src_tensor, src_postion, turns_tensor):
        encode = self.word_embedding(
            src_tensor) + self.pos_embedding(src_postion) + self.turn_embedding(turns_tensor)

        slf_attn_mask = common.get_attn_key_pad_mask(src_tensor, src_tensor)
        non_pad_mask = common.get_non_pad_mask(src_tensor)

        enc_output = self.enc(encode, slf_attn_mask, non_pad_mask)

        return enc_output

    def decode(self, tgt_tensor, src_tensor, enc_output):
        dec_output = self.word_embedding(tgt_tensor)
        dec_output = self.droupout(dec_output)

        non_pad_mask = common.get_non_pad_mask(tgt_tensor)
        slf_attn_mask_subseq = common.get_subsequent_mask(tgt_tensor)
        slf_attn_mask_keypad = common.get_attn_key_pad_mask(
            tgt_tensor, tgt_tensor, True)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = common.get_attn_key_pad_mask(
            src_tensor, tgt_tensor)

        dec_output, m_dec_output = self.dec(
            dec_output, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)

        distributes = self.attention(m_dec_output, enc_output)

        return distributes

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, cuda):
        if cuda:
            self.load_state_dict(torch.load(path))
            self.cuda()
        else:
            self.load_state_dict(torch.load(
                path, map_location=lambda storage, loc: storage))
            self.cpu()
