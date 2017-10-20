import torch

import argparse

from const import *


def corpora2idx(sents, ind2idx):
    return [[ind2idx[w] if w in ind2idx else UNK for w in s] for s in sents]

class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[X]: X,
            UNK_WORD: X
        }
        self.idx2word = {
            X: WORD[X],
            UNK: UNK_WORD
        }
        self.idx = 2

    def add(self, ind):
        if self.word2idx.get(ind) is None:
            self.word2idx[ind] = self.idx
            self.idx2word[self.idx] = ind
            self.idx += 1

    def build_idx(self, sents):
        words = [word for sent in sents for word in sent]
        word_count = {w: 0 for w in set(words)}
        for w in words: word_count[w]+=1

        for word, count in word_count.items():
            self.add(word)

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))

class Corpus(object):
    def __init__(self, train_src, valid_src, save_data, max_len=32):
        self._train_src = train_src
        self._valid_src = valid_src
        self._save_data = save_data
        self.max_len = max_len

        self.sent_dict = Dictionary()
        self.tgt_dict = {
            WORD[X]: X,
            WORD[O]: O,
            WORD[BH]: BH,
            WORD[IH]: IH,
            WORD[BW]: BW,
            WORD[IW]: IW
        }

    def parse_train(self):
        src_sents, labels = [], []
        ignore_text = 0
        for sentence in open(self._train_src):
            words, tgts = [], []

            objs = sentence.strip().split("\t")
            if len(objs) > self.max_len:
                ignore_text += 1
                objs = objs[:self.max_len]

            for obj in objs:
                word, tgt = obj.strip().split(SPLIT_TAG)
                words += [word]
                tgts += [tgt]

            src_sents.append(words)
            labels.append(tgts)

        self.sent_dict.build_idx(src_sents)

        self.src_sents = src_sents
        self.labels = labels
        print("length of text more then {} numbers: {}".format(self.max_len, ignore_text))

    def parse_valid(self):
        src_sents, labels = [], []

        for sentence in open(self._valid_src):
            words, tgts = [], []

            objs = sentence.strip().split("\t")
            if len(objs) > self.max_len:
                objs = objs[:self.max_len]

            for obj in objs:
                word, tgt = obj.strip().split(SPLIT_TAG)
                words += [word]
                tgts += [tgt]

            src_sents.append(words)
            labels.append(tgts)


        self.valid_src_sents = src_sents
        self.valid_labels = labels

    def save(self):
        data = {
            'trains_score': self.trains_score(),
            'max_len': self.max_len,
            'tag_size': len(self.tgt_dict),
            'dict': {
                'src': self.sent_dict.word2idx,
                'vocab_size': len(self.sent_dict),
                'tgt': self.tgt_dict
            },
            'train': {
                'src': corpora2idx(self.src_sents, self.sent_dict.word2idx),
                'label': corpora2idx(self.labels, self.tgt_dict),
            },
            'valid': {
                'src': corpora2idx(self.valid_src_sents, self.sent_dict.word2idx),
                'label': corpora2idx(self.valid_labels, self.tgt_dict),
            }
        }

        torch.save(data, self._save_data)
        print('Finish dumping the corora data to file - [{}]'.format(self._save_data))
        print('words length - [{}]'.format(len(self.sent_dict)))

    def trains_score(self):
        A = {
          'oH':0,
          'oW':0,
          'oo':0,
          'Hh':1.0,
          'hh':0,
          'ho':0,
          'hW':0,
          'Ww':1.0,
          'ww':0,
          'wo':0,
          'wH':0,
        }
        for label in self.labels:
            for t in range(len(label) - 1):
                key = label[t] + label[t+1]
                assert key in A
                A[key] += 1.0

        ts = dict()
        ts['oH'] = A['oH'] / (A['oH'] + A['oW'] + A['oo'])
        ts['oW'] = A['oW'] / (A['oH'] + A['oW'] + A['oo'])
        ts['oo'] = A['oo'] / (A['oH'] + A['oW'] + A['oo'])

        ts['hh'] = A['hh'] / (A['hh'] + A['ho'] + A['hW'])
        ts['ho'] = A['ho'] / (A['hh'] + A['ho'] + A['hW'])
        ts['hW'] = A['hW'] / (A['hh'] + A['ho'] + A['hW'])

        ts['ww'] = A['ww'] / (A['ww'] + A['wo'] + A['wH'])
        ts['wo'] = A['wo'] / (A['ww'] + A['wo'] + A['wH'])
        ts['wH'] = A['wH'] / (A['ww'] + A['wo'] + A['wH'])

        ts["Hh"] = 1.
        ts["Ww"] = 1.

        return ts

    def process(self):
        self.parse_train()
        self.parse_valid()
        self.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bilstm crf nre')
    parser.add_argument('--train-src', type=str, required=True,
                        help='train file')
    parser.add_argument('--save-data', type=str, required=True,
                        help='path to save processed data')
    parser.add_argument('--valid-src', type=str, default=None,
                        help='valid file')
    parser.add_argument('--max-lenth', type=int, default=32,
                        help='max length left of sentence [default: 32]')
    args = parser.parse_args()
    corpus = Corpus(args.train_src, args.valid_src, args.save_data, args.max_lenth)
    corpus.process()
