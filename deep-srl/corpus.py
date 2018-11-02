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


class Labels(Dictionary):
    def __init__(self):
        word2idx = {
            WORD[PAD]: PAD
        }
        super().__init__(word2idx=word2idx, idx_num=len(word2idx))

    def __call__(self, labels):
        _labels = set([l for label in labels for l in label])
        for label in _labels:
            self._add(label)


class Corpus(object):
    def __init__(self, path, save_data, word_max_len=32, label_tag=3):

        self.train = os.path.join(path, "train")
        self.valid = os.path.join(path, "testa")
        self._save_data = save_data
        self._label_tag = label_tag
        self.coutinue_tag = '-DOCSTART-'

        self.w = Words()
        self.l = Labels()
        self.word_max_len = word_max_len

        self.save()

    def parse_data(self, inf, is_train=True):
        sents, labels = [], []
        _words, _labels = [], []
        for sentence in open(inf):
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

        out_of_range_sents = 0
        for index, words in enumerate(sents):
            if len(words) > self.word_max_len:
                out_of_range_sents += 1
                sents[index] = words[:self.word_max_len]
                labels[index] = labels[index][:self.word_max_len]

        if is_train:
            self.w(sents)
            self.l(labels)
            self.train_sents = sents
            self.train_labels = labels
        else:
            self.valid_sents = sents
            self.valid_labels = labels

        print(f"parse down, out of range sents - {out_of_range_sents}")

    def save(self):
        self.parse_data(self.train)
        self.parse_data(self.valid, False)
        data = {
            'word_max_len': self.word_max_len,
            'dict': {
                'word': self.w.word2idx,
                'word_size': len(self.w),
                'label': self.l.word2idx,
                'label_size': len(self.l),
            },
            'train': {
                'word': word2idx(self.train_sents, self.w.word2idx),
                'label': word2idx(self.train_labels, self.l.word2idx)
            },
            'valid': {
                'word': word2idx(self.valid_sents, self.w.word2idx),
                'label': word2idx(self.valid_labels, self.l.word2idx)
            }
        }

        torch.save(data, self._save_data)
        print('Finish dumping the data to file - [{}]'.format(self._save_data))
        print('words length - [{}]'.format(len(self.w)))
        print('labels - [{}]'.format(self.l.word2idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="./data")
    parser.add_argument('--save_data', type=str, default="data/corpus.pt")
    parser.add_argument('--word_max_lenth', type=int, default=40)
    args = parser.parse_args()
    Corpus(args.file_path, args.save_data, args.word_max_lenth)
