import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import re

from const import *
from model import BiLSTM_Cut

import time

class Segment(object):
    def __init__(self, model_source="model", cuda=False):
        self.torch = torch.cuda if cuda else torch
        self.cuda = cuda
        if self.cuda:
            model_source = torch.load(model_source)
        else:
            model_source = torch.load(model_source, map_location=lambda storage, loc: storage)

        self.src_dict = model_source["src_dict"]
        self.trains_score = model_source["trains_score"]
        self.args = args = model_source["settings"]

        model = BiLSTM_Cut(args)
        model.load_state_dict(model_source['model'])

        if self.cuda:
            model = model.cuda()
            model.prob_projection = nn.Softmax().cuda()
        else:
            model = model.cpu()
            model.prob_projection = nn.Softmax().cpu()

        self.model = model.eval()

    def text2tensor(self, text):
        ids = [self.src_dict[w] if w in self.src_dict else UNK for w in text]
        ids = Variable(torch.LongTensor(ids).unsqueeze(0), volatile=True)
        if self.cuda:
            ids = ids.cuda()
        return ids

    def viterbi(self, nodes):
        paths = {WORD[B]: nodes[0][WORD[B]], WORD[S]: nodes[0][WORD[S]]}

        for w_step in range(1, len(nodes)):
            _path = paths.copy()
            paths = {}

            sub_paths = {}
            for code, score in nodes[w_step].items():
                for last_code, last_score in _path.items():
                    if last_code[-1] + code in self.trains_score:
                        sub_paths[last_code+code] = last_score*score*self.trains_score[last_code[-1] + code]

            sorted_sub_path = sorted(sub_paths.items(),
                                key=lambda path: path[1],
                                reverse=True)

            best_path, best_score = sorted_sub_path[0]
            paths[best_path] = best_score

        sorted_path = sorted(paths.items(),
                            key=lambda path: path[1],
                            reverse=True)
        best_path, _ = sorted_path[0]
        return best_path

    def text_cut(self, text):
        if text:
            text_len = len(text)
            tensor = self.text2tensor(text)
            pre = self.model.prob_projection(self.model(tensor))
            nodes = [dict(zip([WORD[B], WORD[M], WORD[E], WORD[S]], each[1:])) for each in pre.data]

            tags = self.viterbi(nodes)
            words = []
            for i in range(len(text)):
                if tags[i] in [WORD[B], WORD[S]]:
                    words.append(text[i])
                else:
                    words[-1] += text[i]

            return words
        else:
            return []

    def sentence_cut(self, sentence):
        not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
        result = []
        start = 0
        for seg_sign in not_cuts.finditer(sentence):
            result.extend(self.text_cut(sentence[start:seg_sign.start()]))
            result.append(sentence[seg_sign.start():seg_sign.end()])
            start = seg_sign.end()
        result.extend(self.text_cut(sentence[start:]))
        return result

if __name__ == "__main__":
    import time
    sg = Segment()
    print(sg.sentence_cut("ngram是自然语言处理中一个非常重要的概念，通常在NLP中，人们基于一定的语料库，可以利用ngram来预计或者评估一个句子是否合理。另外一方面，ngram的另外一个作用是用来评估两个字符串之间的差异程度。这是模糊匹配中常用的一种手段。本文将从此开始，进而向读者展示ngram在自然语言处理中的各种powerful的应用"))
