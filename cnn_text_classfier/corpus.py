import os
_root = os.path.normpath("%s/.." % os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(_root)

import xlrd
import torch
import logging
import json
import time
import random

from word_filter import *
from segmenter import Jieba

EOS_token = "EOS"

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {0: EOS_token}
        self.idx = 1

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.word2count[word] = 1
            self.idx += 1
        else:
            self.word2count[word] += 1

    def discard(self, t=1e-3):
        self.idx2word = {0: EOS_token}
        self.word2idx = {}
        self.idx = 1

        for word, count in self.word2count.items():
            dis_ture = (1 - (t/count) ** 0.5) >= random.random()
            if dis_ture: continue
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))

class LabelDic(object):
    def __init__(self):
        self.label2idx = {}
        self.idx2label = {}
        self.idx = 0

    def add_label(self, label):
        self.label2idx[label] = self.idx
        self.idx2label[self.idx] = label
        self.idx += 1

    def has_label(self, label):
        if label in self.label2idx:
            return True
        return False

    def __len__(self):
        return len(self.label2idx)

class Corpus(object):
    def __init__(self, path="./data", max_len=20):
        self.jb = Jieba("./segmenter_dicts")
        self.swf = StopwordFilter("./segmenter_dicts/stopwords.txt")
        self.dictionary = Dictionary()
        self.train = os.path.join(path, "train.xlsx")
        self.test = os.path.join(path, "test.xlsx")
        self.label = LabelDic()
        self.max_len = max_len

    def save(self, path="./model"):
        with open(os.path.join(path, "dict"), 'w') as f:
            f.write(json.dumps(self.dictionary.word2idx))
        f.close()

        with open(os.path.join(path, "label"), 'w') as f:
            f.write(json.dumps(self.label.idx2label))
        f.close()

    def get_testdata(self, docs=evl_docs):
        length = len(docs)
        test_data, test_ids = torch.LongTensor(length*self.max_len).zero_(), torch.LongTensor(length).zero_()
        for index, doc in enumerate(docs):
            token = 0
            v1, v2 = doc[0].strip(), doc[1].strip()
            words = [e[0] for e in self.jb.segment(v1) if self.swf.filter(e[0]) and e[0] in self.dictionary.word2idx]
            if len(words) > self.max_len: words = words[:self.max_len]
            for word in words:
                test_data[index*self.max_len+token] = self.dictionary.word2idx[word]
                token += 1
            test_ids[index] = self.label.label2idx[v2]

        return test_data.view(-1, self.max_len), test_ids

    def get_data(self):
        """
        - Add words to the dictionary
        - Tokenize the file content
        """
        start_time = time.time()
        try:
            excel_book = xlrd.open_workbook(self.train)
            logging.info("open {}".format(self.train))
            table = excel_book.sheets()[0]
        except Exception as e:
            logging.error("open {} failed: {}".format(self.train, e))

        # step one: add words
        for i in range(table.nrows):
            value = table.row_values(i)
            if i == 0 or len(value) < 3 or not isinstance(value[1], str) or not isinstance(value[2], str): continue
            v1, v2 = value[1].strip(), value[2].strip()
            words = [e[0] for e in self.jb.segment(v2) if self.swf.filter(e[0])]
            for word in words: self.dictionary.add_word(word)
            if self.label.has_label(v1): continue
            self.label.add_label(v1)

        # step two: discard low tf words
        self.dictionary.discard()

        # step three: count doc lines
        docs = 0
        for i in range(table.nrows):
            if i == 0 or len(value) < 3 or not isinstance(value[1], str) or not isinstance(value[2], str): continue
            if len([e[0] for e in self.jb.segment(v2) if self.swf.filter(e[0]) and e[0] in self.dictionary.word2idx]) == 0: continue
            docs += 1

        train_ids, label_ids = torch.LongTensor(docs*self.max_len).zero_(), torch.LongTensor(docs).zero_()

        # step four: tokens to tensor
        docs = 0
        for i in range(table.nrows):
            value = table.row_values(i)
            if len(value) < 3 or not isinstance(value[1], str) or not isinstance(value[2], str): continue
            token = 0
            v1, v2 = value[1].strip(), value[2].strip()
            words = [e[0] for e in self.jb.segment(v2) if self.swf.filter(e[0]) and e[0] in self.dictionary.word2idx]
            if len(words) == 0: continue
            if len(words) > self.max_len: words = words[:self.max_len]
            for word in words:
                train_ids[docs*self.max_len+token] = self.dictionary.word2idx[word]
                token += 1
            label_ids[docs] = self.label.label2idx[v1]
            docs += 1
        print('=' * 100)
        print("Load corpus done, tokens-[{}], labels-[{}], docs-[{}], time cost-[{:4.2f}]second".format(len(self.dictionary), len(self.label), docs, (time.time()-start_time)))
        print('=' * 100)
        return train_ids.view(-1, self.max_len), label_ids




