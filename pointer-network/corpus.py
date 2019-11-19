import torch
import numpy as np

import argparse
import logging

import common
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
    def __init__(self, save_data=f"{DATAPATH}/corpus.pt", min_word_count=1, max_len=184):
        self._save_data = save_data
        self.min_word_count = min_word_count
        self.max_len = max_len
        self.dict = Dictionary()
        self.parse()
        self.save()

    def parse(self):
        def parse_file(inf):
            src_texts, src_turn, tgt_indexs, tgt_texts = [], [], [], []
            with open(inf, encoding="utf8") as contexts:
                for line in contexts:
                    query1, query2, query3, target = line.strip().split("\t\t")

                    q1_words = common.split_char(query1)
                    turn1 = [1]*(len(q1_words))
                    q2_words = common.split_char(query2)
                    turn2 = [2]*(len(q2_words)+1)
                    q3_words = common.split_char(query3)
                    turn3 = [3]*(len(q3_words))

                    q_words = q1_words + q2_words + [WORD[EOS]] + q3_words
                    turns = turn1 + turn2 + turn3
                    if len(q_words) > self.max_len:
                        continue

                    assert len(q_words) == len(turns)
                    src_texts.append(q_words)
                    src_turn.append(turns)

                    tgt_words = common.split_char(target)
                    new_tgt_words = []
                    for word in tgt_words:
                        if word in q_words:
                            new_tgt_words.append(word)

                    tgt_texts.append([WORD[BOS]]+new_tgt_words)

                    new_tgt_words = new_tgt_words + [WORD[EOS]]
                    t_index = common.find_text_index(q_words, new_tgt_words)

                    tgt_indexs.append(t_index)

                return src_texts, src_turn, tgt_indexs, tgt_texts

        src_texts, src_turn, tgt_indexs, tgt_texts = parse_file(
            f"{DATAPATH}/corpus")
        print(
            f"Ignored word counts - {self.dict(src_texts, self.min_word_count)}")

        src_texts = np.asarray(common.texts2idx(src_texts, self.dict.word2idx))
        src_turn = np.asarray(src_turn)
        tgt_indexs = np.asarray(tgt_indexs)
        tgt_texts = np.asarray(common.texts2idx(tgt_texts, self.dict.word2idx))

        assert src_texts.shape == src_turn.shape
        assert tgt_indexs.shape == tgt_texts.shape

        index = np.arange(tgt_texts.shape[0])
        np.random.shuffle(index)
        src_texts = src_texts[index]
        src_turn = src_turn[index]
        tgt_indexs = tgt_indexs[index]
        tgt_texts = tgt_texts[index]

        self.src_texts_train = src_texts[2000:]
        self.src_turn_train = src_turn[2000:]
        self.tgt_indexs_train = tgt_indexs[2000:]
        self.tgt_texts_train = tgt_texts[2000:]

        self.src_texts_test = src_texts[:2000]
        self.src_turn_test = src_turn[:2000]
        self.tgt_indexs_test = tgt_indexs[:2000]
        self.tgt_texts_test = tgt_texts[:2000]

    def save(self):
        data = {
            'word2idx':  self.dict.word2idx,
            'max_len':  self.max_len,
            'train': {
                'src_texts': self.src_texts_train,
                'src_turn': self.src_turn_train,
                'tgt_indexs': self.tgt_indexs_train,
                'tgt_texts':  self.tgt_texts_train,
            },
            'valid': {
                'src_texts': self.src_texts_test,
                'src_turn': self.src_turn_test,
                'tgt_indexs': self.tgt_indexs_test,
                'tgt_texts':  self.tgt_texts_test,
            }
        }

        torch.save(data, self._save_data)
        print(f'corpora length - {len(self.dict)}')


if __name__ == "__main__":
    Corpus()
