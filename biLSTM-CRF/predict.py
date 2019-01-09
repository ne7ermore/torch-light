import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import re

from const import *
from model import BiLSTM_CRF_Size


class Predict(object):
    def __init__(self, model_source, cuda=False):
        self.torch = torch.cuda if cuda else torch
        self.cuda = cuda
        if self.cuda:
            model_source = torch.load(model_source)
        else:
            model_source = torch.load(
                model_source, map_location=lambda storage, loc: storage)

        self.src_dict = model_source["src_dict"]
        self.trains_score = model_source["trains_score"]
        self.args = args = model_source["settings"]

        model = BiLSTM_CRF_Size(args)
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
        paths = {
            WORD[O]: nodes[0][WORD[O]],
            WORD[BH]: nodes[0][WORD[BH]],
            WORD[BW]: nodes[0][WORD[BW]]
        }

        for w_step in range(1, len(nodes)):
            _path = paths.copy()
            paths = {}

            self.sub_paths = {}
            for code, score in nodes[w_step].items():
                for last_code, last_score in _path.items():
                    if last_code[-1] + code in self.trains_score:
                        self.sub_paths[last_code+code] = last_score * \
                            score*self.trains_score[last_code[-1] + code]

            sorted_sub_path = sorted(self.sub_paths.items(),
                                     key=lambda path: path[1],
                                     reverse=True)

            best_path, best_score = sorted_sub_path[0]
            paths[best_path] = best_score

        sorted_path = sorted(paths.items(),
                             key=lambda path: path[1],
                             reverse=True)
        best_path, _ = sorted_path[0]
        return best_path

    def get_size(self, text):
        res = {}
        if text == "":
            return res

        tensor = self.text2tensor(text)
        pre = self.model.prob_projection(self.model(tensor))
        nodes = [dict(zip([WORD[O], WORD[BH], WORD[IH], WORD[BW],
                           WORD[IW]], each[1:])) for each in pre.data]

        tags = self.viterbi(nodes)

        height, weight = [], []
        for index, word in enumerate(text):
            if tags[index] in [WORD[BH], WORD[IH]]:
                height.append(word)

            if tags[index] in [WORD[BW], WORD[IW]]:
                weight.append(word)

        if len(height) != 0:
            res["height"] = "".join(height)
        if len(weight) != 0:
            res["weight"] = "".join(weight)

        return res
