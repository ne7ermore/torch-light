import torch
import re
import os
import sys
import json
import collections
import random

import numpy as np

import const
import utils


def word2idx(words, word2idx):
    return [word2idx[w] if w in word2idx else const.UNK for w in words]


class Dictionary(object):

    dots = ['.', '?', '!', "。", '？', '！', ",", "，"]

    def __init__(self):
        self.word2idx = {
            const.WORD[const.PAD]: const.PAD,
            const.WORD[const.UNK]: const.UNK,
        }
        self.idx = len(self.word2idx)
        self.add_dot()
        self.word_count = collections.defaultdict(int)

    def add_dot(self):
        for dot in self.dots:
            self.word2idx[dot] = self.idx
            self.idx += 1

    def add(self, word):
        chars = ''.join(c for c in word if utils.is_chinese_char(ord(c)))
        if chars:
            self.word_count[word] += 1

    def parse(self, min_count=0):
        ignored_word_count = 0
        for word, count in self.word_count.items():
            if count <= min_count:
                ignored_word_count += 1
                continue

            if self.word2idx.get(word) is None:
                self.word2idx[word] = self.idx
                self.idx += 1

        return ignored_word_count


class Corpus(object):
    def __init__(self, inp_data=const.DATAPATH, load_w2v=None,
                 save_data=os.path.join(const.DATAPATH, "corpus.pt")):

        self.save_data = save_data
        self.load_w2v = load_w2v
        self.inp_data = inp_data
        self.word = Dictionary()
        self.char = Dictionary()
        self.load_files()
        self.save()

    def load_files(self):
        data_path = f"{self.inp_data}/data/train"
        for _, _, files in os.walk(data_path):
            for inf in files:
                if inf.endswith("conll"):
                    for doc in utils.load_file(f"{data_path}/{inf}"):
                        tokens = doc.tokens
                        [self.word.add(w) for w in tokens]
                        [self.char.add(c) for w in tokens for c in w]

        print(f'ignored word count - {self.word.parse(2)}')
        print(f'ignored char count - {self.char.parse(2)}')

    def save(self):
        data = {
            'word2idx': self.word.word2idx,
            'char2idx': self.char.word2idx,
        }
        if self.load_w2v is not None:
            data["wordW"], data["charW"] = utils.load_pre_w2c(
                self.load_w2v, self.word.word2idx, self.char.word2idx)

        torch.save(data, self.save_data)
        print(f'char length - {len(self.char.word2idx)}')
        print(f'word length - {len(self.word.word2idx)}')
        print(f'Finish dumping the data to file - {self.save_data}')


if __name__ == "__main__":
    Corpus()
