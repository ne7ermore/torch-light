import re

import torch

from const import *


def word2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


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
    def __init__(self, fuel_path, save_data, max_len, min_word_count=1):
        self._fuel_path = fuel_path
        self._save_data = save_data
        self.max_len = max_len
        self._min_word_count = min_word_count
        self.src_dict = Dictionary()
        self.tgt_dict = Dictionary()

    def parse(self):
        def gather_file(file_, max_len):
            en_sents, he_sents, en_cut_count, he_cut_count = [], [], 0, 0

            for sentences in open(file_):
                en_, he_ = [normalizeString(s)
                            for s in sentences.strip().split('\t')]

                en_ws = [word for word in en_.strip().split()]
                he_ws = [word for word in he_.strip().split()]

                if len(en_ws) > max_len:
                    en_cut_count += 1
                    en_ws = en_ws[:max_len]
                en_sents.append(en_ws + [WORD[EOS]])

                if len(he_ws) > max_len:
                    he_cut_count += 1
                    he_ws = he_ws[:max_len]
                he_sents.append(he_ws + [WORD[EOS]])

            return he_sents, en_sents, he_cut_count, en_cut_count

        max_len = self.max_len - 1
        src_train, tgt_train, he_cut_count, en_cut_count = gather_file(
            'data/train', max_len)
        src_valid, tgt_valid, _, _ = gather_file('data/test', max_len)

        print(
            "English data`s length out of range numbers - [{}]".format(en_cut_count))
        print(
            "He data`s length out of range numbers - [{}]".format(he_cut_count))

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
            'max_len': self.max_len,
            'dict': {
                'src': self.src_dict.word2idx,
                'src_size': len(self.src_dict),
                'tgt': self.tgt_dict.word2idx,
                'tgt_size': len(self.tgt_dict),
                'src_id2w': {v: k for k, v in self.src_dict.word2idx.items()},
                'tgt_id2w': {v: k for k, v in self.tgt_dict.word2idx.items()}
            },
            'train': {
                'data': word2idx(self.src_train, self.src_dict.word2idx),
                'label': word2idx(self.tgt_train, self.tgt_dict.word2idx),
            },
            'valid': {
                'data': word2idx(self.src_valid, self.src_dict.word2idx),
                'label': word2idx(self.tgt_valid, self.tgt_dict.word2idx),
            }
        }

        torch.save(data, self._save_data)
        print('src word length - [{}]'.format(len(self.src_dict)))
        print('tgt word length - [{}]'.format(len(self.tgt_dict)))

    def process(self):
        self.parse()
        self.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='drsm')
    parser.add_argument('--save_data', type=str, default='data/corpus')
    parser.add_argument('--fuel_path', type=str, default='data/')
    parser.add_argument('--max_len', type=int, default=17)
    parser.add_argument('--min_word_count', type=int, default=1)
    args = parser.parse_args()

    corpus = Corpus(args.fuel_path, args.save_data,
                    args.max_len, args.min_word_count)
    corpus.process()
