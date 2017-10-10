"""
Seq2seq modules.
Heavily borrowed from jadore801120/attention-is-all-you-need-pytorch.
Please check the following link for code:
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py

Some optimizations in coding and modules. eq: Scaled Dot-Production Attention, etc...
"""
import torch
import torch.nn as nn
import torch.nn.init as init

from layers import EncoderLayer, DecoderLayer
from model_utils import *
import const

class Encoder(nn.Module):
    def __init__(self, n_src_vocab, n_max_seq, n_layers=6, n_head=8,
            emb_dim=512, d_model=512, d_inner_hid=2048, dropout=0.1):
        super().__init__()
        n_position = n_max_seq + 1
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.n_position = n_position

        self.position_enc = nn.Embedding(n_position, emb_dim, padding_idx=const.PAD)
        self.src_word_emb = nn.Embedding(n_src_vocab, emb_dim, padding_idx=const.PAD)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner_hid, n_head, dropout=dropout) for _ in range(n_layers)])

        self._init_weight()

    def _init_weight(self):
        self.position_enc.weight.data = position_encoding_init(
                                        self.n_position, self.emb_dim)
        self.position_enc.weight.requires_grad = False

    def forward(self, src_seq, src_pos):
        enc_input = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_outputs, enc_slf_attns = [], []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
            enc_outputs += [enc_output]
            enc_slf_attns += [enc_slf_attn]
        return enc_outputs, enc_slf_attns

class Decoder(nn.Module):
    def __init__(self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,
            emb_dim=512, d_model=512, d_inner_hid=2048, dropout=0.1):
        super().__init__()
        n_position = n_max_seq + 1
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.n_position = n_position

        self.position_enc = nn.Embedding(n_position, emb_dim, padding_idx=const.PAD)
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, emb_dim, padding_idx=const.PAD)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, dropout=dropout)
            for _ in range(n_layers)])

        self._init_weight()

    def _init_weight(self):
        self.position_enc.weight.data = position_encoding_init(
                                        self.n_position, self.emb_dim)
        self.position_enc.weight.requires_grad = False

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_outputs):
        dec_input = self.tgt_word_emb(tgt_seq)
        dec_input += self.position_enc(tgt_pos)
        dec_outputs, dec_slf_attns, dec_enc_attns = [], [], []

        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        dec_output = dec_input
        for dec_layer, enc_output in zip(self.layer_stack, enc_outputs):
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            dec_outputs += [dec_output]
            dec_slf_attns += [dec_slf_attn]
            dec_enc_attns += [dec_enc_attn]

        return dec_outputs, dec_slf_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=6,
                 n_head=8, emb_dim=512, d_model=512, d_inner_hid=2048,
                 dropout=0.1, proj_share_weight=True, embs_share_weight=False):
        super().__init__()
        self.encoder = Encoder(n_src_vocab, n_max_seq,
                               n_layers=n_layers,
                               n_head=n_head,
                               emb_dim=emb_dim,
                               d_model=d_model,
                               d_inner_hid=d_inner_hid,
                               dropout=dropout)

        self.decoder = Decoder(n_tgt_vocab, n_max_seq,
                               n_layers=n_layers,
                               n_head=n_head,
                               emb_dim=emb_dim,
                               d_model=d_model,
                               d_inner_hid=d_inner_hid,
                               dropout=dropout)

        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

        # paper: as well as the embedding layers, produce outputs of dimension d_model = 512
        assert d_model == emb_dim, 'To facilitate the residual connections, the dimensions of all module output shall be the same.'

        if proj_share_weight:
            # Share the weight matrix between tgt word embedding/projection
            assert d_model == emb_dim
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight
        else:
            init.xavier_normal(self.tgt_word_proj.weight)

        if embs_share_weight:
            # Share the weight matrix between src/tgt word embeddings
            # assume the src/tgt word vec size are the same
            assert n_src_vocab == n_tgt_vocab
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_outputs, enc_slf_attns = self.encoder(src_seq, src_pos)
        dec_outputs, dec_slf_attns, dec_enc_attns = self.decoder(tgt_seq, tgt_pos, src_seq, enc_outputs)
        dec_output = dec_outputs[-1]

        seq_logit = self.tgt_word_proj(dec_output) # bsz*(tgt.n_max_seq-1)*n_tgt_vocab

        return seq_logit.view(-1, seq_logit.size(2))
