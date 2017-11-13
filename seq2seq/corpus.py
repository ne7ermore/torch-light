import torch

import argparse
import logging

from utils import corpora2idx, normalizeString
from const import *

class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[BOS]: BOS,
            WORD[EOS]: EOS,
            WORD[PAD]: PAD,
            WORD[UNK]: UNK
        }
        self.idx = 4

    def add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def __call__(self, sents, min_count):
        words = [word for sent in sents for word in sent]
        word_count = {w: 0 for w in set(words)}
        for w in words: word_count[w]+=1

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
    def __init__(self, save_data, max_len=20, min_word_count=1):
        self._save_data = save_data
        self._max_len = max_len
        self._min_word_count = min_word_count
        self.src_sents = None
        self.tgt_sents = None
        self.src_valid_sents = None
        self.tgt_valid_sents = None
        self.src_dict = Dictionary()
        self.tgt_dict = Dictionary()

    def parse(self):
        def gather_file(file_, max_len):
            en_sents, fra_sents, en_cut_count, fra_cut_count = [], [], 0, 0

            for sentences in open(file_):
                en_, fra_ = [normalizeString(s) for s in sentences.strip().split('\t')]

                en_ws = [word for word in en_.strip().split()]
                fra_ws = [word for word in fra_.strip().split()]

                if len(en_ws) > max_len:
                    en_cut_count += 1
                    en_ws = en_ws[:max_len]
                en_sents.append([WORD[BOS]] + en_ws + [WORD[EOS]])

                if len(fra_ws) > max_len:
                    fra_cut_count += 1
                    fra_ws = fra_ws[:max_len]
                fra_sents.append([WORD[BOS]] + fra_ws + [WORD[EOS]])

            return fra_sents, en_sents, fra_cut_count, en_cut_count

        max_len = self._max_len - 2
        src_train, tgt_train, fra_cut_count, en_cut_count = gather_file('data/train', max_len)
        src_valid, tgt_valid, _, _ = gather_file('data/test', max_len)

        print("English data`s length out of range numbers - [{}]".format(en_cut_count))
        print("French data`s length out of range numbers - [{}]".format(fra_cut_count))

        src_ignore = self.src_dict(src_train, self._min_word_count)
        tgt_ignore = self.tgt_dict(tgt_train, self._min_word_count)
        if src_ignore != 0:
            print("Ignored src word counts - [{}]".format(src_ignore))
        if tgt_ignore != 0:
            print("Ignored tgt word counts - [{}]".format(tgt_ignore))

        self.src_train = src_train
        self.tgt_train = tgt_train
        self.src_valid = src_valid
        self.tgt_valid = tgt_valid

    def save(self):
        data = {
            'max_word_len': self._max_len,
            'dict': {
                'src': self.src_dict.word2idx,
                'src_size': len(self.src_dict),
                'tgt': self.tgt_dict.word2idx,
                'tgt_size': len(self.tgt_dict)
            },
            'train': {
                'src': corpora2idx(self.src_train, self.src_dict.word2idx),
                'tgt': corpora2idx(self.tgt_train, self.tgt_dict.word2idx)
            },
            'valid': {
                'src': corpora2idx(self.src_valid, self.src_dict.word2idx),
                'tgt': corpora2idx(self.tgt_valid, self.tgt_dict.word2idx)
            }
        }

        torch.save(data, self._save_data)
        print('src corpora length - [{}] | target corpora length - [{}]'.format(len(self.src_dict), len(self.tgt_dict)))

    def process(self):
        self.parse()
        self.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='seq2sqe corpora')
    parser.add_argument('--save-data', type=str, default='data/seq2seq.pt',
                        help='path to save processed data')
    parser.add_argument('--max-lenth', type=int, default=20,
                        help='max length of sentence')
    parser.add_argument('--min-word-count', type=int, default=1,
                        help='min corpora count to discard')
    args = parser.parse_args()
    corpus = Corpus(args.save_data, args.max_lenth, args.min_word_count)
    corpus.process()
