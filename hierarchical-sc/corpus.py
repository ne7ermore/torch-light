import os
import torch
import math

import pandas as pd

from const import *


def word2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]


class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK,
            WORD[BOS]: BOS,
            WORD[EOS]: EOS
        }
        self.idx = len(self.word2idx)

    def add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def __call__(self, sents, min_count):
        words = [word for sent in sents for word in sent]
        word_count = {w: 0 for w in set(words)}
        for w in words:
            word_count[w] += 1

        ignored_word_count = 0
        for word, count in word_count.items():
            if count <= min_count:
                ignored_word_count += 1
                continue
            self.add(word)

        return ignored_word_count

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))


class Corpus(object):
    def __init__(self, max_ori_len=128, max_sum_len=15, min_word_count=1):
        self.dict = Dictionary()
        self.max_ori_len = max_ori_len
        self.max_sum_len = max_sum_len
        self._min_word_count = min_word_count

        self.parse_data("data/test.csv", False)
        self.parse_data("data/train.csv")
        self.save()

    def parse_data(self, _file, is_train=True):
        def cut(x, list, ignore, max_len, is_summ):
            if isinstance(x, float) and math.isnan(x):
                if is_summ:
                    list.append(WORD[EOS])
                else:
                    list.append("")
            else:
                x = x.split()
                if len(x) > max_len:
                    x = x[:max_len]
                    ignore[0] += 1

                if is_summ:
                    x += [WORD[EOS]]

                list.append(x)

        origins, summurys = [], []
        ignore_ori_nums = [0]
        ignore_sum_nums = [0]

        df = pd.read_csv(_file)

        df["original"].apply(cut, args=(
            origins, ignore_ori_nums, self.max_ori_len, False))
        df["summary"].apply(cut, args=(
            summurys, ignore_sum_nums, self.max_sum_len, True))

        if is_train:
            ori_ignore = self.dict(origins + summurys, self._min_word_count)
            self.train_origins = origins
            self.train_summurys = summurys
            self.train_labels = df["score"].values - 1

            print("Ignored origin counts - [{}]".format(ori_ignore))
            print(
                'Train data - ignore original lines - [{}]'.format(ignore_ori_nums[0]))
            print(
                'Train data - ignore summary lines - [{}]'.format(ignore_sum_nums[0]))
        else:
            self.test_origins = origins
            self.test_summurys = summurys
            self.test_labels = df["score"].values - 1
            print(
                'Test data - ignore original lines - [{}]'.format(ignore_ori_nums[0]))
            print(
                'Test data - ignore summary lines - [{}]'.format(ignore_sum_nums[0]))

    def save(self):
        data = {
            'max_ori_len': self.max_ori_len,
            'max_sum_len': self.max_sum_len + 1,
            'dict': {
                'dict': self.dict.word2idx,
                'dict_size': len(self.dict),
            },
            'train': {
                'original': word2idx(self.train_origins, self.dict.word2idx),
                'summary': word2idx(self.train_summurys, self.dict.word2idx),
                'label': self.train_labels
            },
            'test': {
                'original': word2idx(self.test_origins, self.dict.word2idx),
                'summary': word2idx(self.test_summurys, self.dict.word2idx),
                'label': self.test_labels
            }
        }

        torch.save(data, "data/corpus")
        print('dict length - [{}]'.format(len(self.dict)))


if __name__ == "__main__":
    Corpus()
