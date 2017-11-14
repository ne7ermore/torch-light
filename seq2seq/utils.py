# borrowed from https://github.com/jadore801120/attention-is-all-you-need-pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import re

import const

def position(n_position, d_model):
    position_enc = np.array([[pos / np.power(10000, 2*i/d_model) for i in range(d_model)] for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.sin(position_enc[1:, 1::2])

    return torch.from_numpy(position_enc).float()

def get_attn_padding_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    bsz, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(const.PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(bsz, len_q, len_k)
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

def corpora2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else const.UNK for w in s] for s in sents]

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
