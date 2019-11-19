import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import re

import const


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(const.PAD).type(torch.float).unsqueeze(-1)


def get_padding_mask(x):
    return x.eq(0)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q, byte=False):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(const.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
    if byte:
        return padding_mask.byte()
    return padding_mask


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)

    return subsequent_mask


def is_chinese_char(c):
    if ((c >= 0x4E00 and c <= 0x9FFF) or
            (c >= 0x3400 and c <= 0x4DBF) or
            (c >= 0x20000 and c <= 0x2A6DF) or
            (c >= 0x2A700 and c <= 0x2B73F) or
            (c >= 0x2B740 and c <= 0x2B81F) or
            (c >= 0x2B820 and c <= 0x2CEAF) or
            (c >= 0xF900 and c <= 0xFAFF) or
            (c >= 0x2F800 and c <= 0x2FA1F)):
        return True
    return False


def split_char(text):
    text = "".join([w for w in text.split()])
    step, words = 0, []
    un_chinese = ""
    while step < len(text):
        if is_chinese_char(ord(text[step])):
            words.append(text[step])
            step += 1
        else:
            while step < len(text):
                if is_chinese_char(ord(text[step])):
                    words.append(un_chinese.lower())
                    un_chinese = ""
                    break
                un_chinese += text[step]
                step += 1
    if un_chinese:
        return words + [un_chinese.lower()]
    return words


def texts2idx(texts, word2idx):
    return [[word2idx[word] if word in word2idx else const.UNK for word in text] for text in texts]


def find_index(text, word):
    stop_index = text.index(const.WORD[const.EOS])
    if word in text[stop_index:]:
        idx = text.index(word, stop_index)
    else:
        idx = text.index(word)
    text[idx] = "@@@"
    return idx


def find_text_index(q_words, new_tgt_words):
    word_map, q_words = {}, q_words.copy()
    t_index = np.zeros(len(new_tgt_words), dtype=int)
    for index, word in enumerate(new_tgt_words):
        if word in q_words:
            pointer = find_index(q_words, word)
            t_index[index] = pointer
            word_map[word] = pointer
        elif word in word_map:
            t_index[index] = word_map[word]
        else:
            raise Exception(
                f"invalid word {word} from {''.join(q_words)} {''.join(new_tgt_words)}")
    return t_index
