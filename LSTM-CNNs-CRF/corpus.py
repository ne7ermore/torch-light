import torch

import logging
import argparse
import os
import copy
import re

from const import *


def normalizeString(s):
    s = s.lower().strip()
    try:
        float(s)
        return "@"
    except:
        return s


def word2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]


def char2idx(sents, char2idx):
    return [[[char2idx[c] if c in char2idx else UNK for c in word] for word in sent] for sent in sents]


class Dictionary(object):
    def __init__(self, word2idx={}, idx_num=0):
        self.word2idx = word2idx
        self.idx = idx_num

    def _add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def _convert(self):
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))


class Words(Dictionary):
    def __init__(self):
        word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK
        }
        super().__init__(word2idx=word2idx, idx_num=len(word2idx))

    def __call__(self, sents):
        words = set([word for sent in sents for word in sent])
        for word in words:
            self._add(word)


class Chars(Dictionary):
    def __init__(self):
        word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK
        }
        super().__init__(word2idx=word2idx, idx_num=len(word2idx))

    def __call__(self, sents):
        chars = set([char for sent in sents for word in sent for char in word])
        for char in chars:
            self._add(char)


class Labels(Dictionary):
    def __init__(self):
        word2idx = {
            TAG[PAD]: PAD,
            TAG[START]: START,
            TAG[STOP]: STOP
        }
        super().__init__(word2idx=word2idx, idx_num=len(word2idx))

    def __call__(self, labels):
        _labels = set([l for label in labels for l in label])
        for label in _labels:
            self._add(label)


class Corpus(object):
    def __init__(self, path, save_data,
                 word_max_len=32, char_max_len=8, label_tag=3):

        self.train = os.path.join(path, "train")
        self.valid = os.path.join(path, "testa")
        self._save_data = save_data
        self._label_tag = label_tag
        self.coutinue_tag = '-DOCSTART-'

        self.w = Words()
        self.c = Chars()
        self.l = Labels()
        self.word_max_len = word_max_len
        self.char_max_len = char_max_len

    def parse_data(self, _file, is_train=True):
        sents, labels = [], []
        _words, _labels = [], []
        for sentence in open(_file):
            if sentence == '\n':
                if len(_words) == 0:
                    continue
                sents.append(_words.copy())
                labels.append(_labels.copy())
                _words, _labels = [], []
                continue
            temp = sentence.strip().split(' ')

            label, word = temp[self._label_tag].strip(), temp[0].strip()
            if word == self.coutinue_tag:
                continue

            _words += [normalizeString(word)]
            _labels += [label]

        out_of_range_sents = out_of_range_words = 0
        dc_sents = copy.deepcopy(sents)

        for index, words in enumerate(sents):
            if len(words) > self.word_max_len:
                out_of_range_sents += 1
                sents[index] = words[:self.word_max_len]
                labels[index] = labels[index][:self.word_max_len]
                dc_sents[index] = words[:self.word_max_len]

            for w_index, word in enumerate(dc_sents[index]):
                if len(word) > self.char_max_len:
                    out_of_range_words += 1
                    dc_sents[index][w_index] = word[:self.char_max_len]

            dc_sents[index] = [[char for char in word]
                               for word in dc_sents[index]]

        if is_train:
            self.w(sents)
            self.c(dc_sents)
            self.l(labels)
            self.train_sents = sents
            self.train_chars = dc_sents
            self.train_labels = labels
        else:
            self.valid_sents = sents
            self.valid_chars = dc_sents
            self.valid_labels = labels

        print("parse down, out of range sents - {}, out of range words - {}".format(
            out_of_range_sents, out_of_range_words))

    def save(self):
        self.parse_data(self.train)
        self.parse_data(self.valid, False)
        data = {
            'word_max_len': self.word_max_len,
            'char_max_len': self.char_max_len,
            'dict': {
                'word': self.w.word2idx,
                'word_size': len(self.w),
                'char': self.c.word2idx,
                'char_size': len(self.c),
                'label': self.l.word2idx,
                'label_size': len(self.l),
            },
            'train': {
                'word': word2idx(self.train_sents, self.w.word2idx),
                'char': char2idx(self.train_chars, self.c.word2idx),
                'label': word2idx(self.train_labels, self.l.word2idx)
            },
            'valid': {
                'word': word2idx(self.valid_sents, self.w.word2idx),
                'char': char2idx(self.valid_chars, self.c.word2idx),
                'label': word2idx(self.valid_labels, self.l.word2idx)
            }
        }

        torch.save(data, self._save_data)
        print('Finish dumping the data to file - [{}]'.format(self._save_data))
        print('words length - [{}]'.format(len(self.w)))
        print('chars length - [{}]'.format(len(self.c)))
        print('labels - [{}]'.format(self.l.word2idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type=str, default="./data",
                        help='file path')
    parser.add_argument('--save-data', type=str, default="corpus.pt",
                        help='path to save processed data')
    parser.add_argument('--word-max-lenth', type=int, default=40,
                        help='max length left of sentence')
    parser.add_argument('--char-max-lenth', type=int, default=10,
                        help='max length left of word')
    args = parser.parse_args()
    corpus = Corpus(args.file_path, args.save_data,
                    args.word_max_lenth, args.char_max_lenth)
    corpus.save()
