import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import jieba.posseg as pseg

from const import *
from model import Model
import common

import time
import copy
import os
import sys


class Predict(object):
    def __init__(self, model_datas=None, model_source=None, cuda=False):
        assert model_datas is not None or model_source is not None, "model and model_source should not be both None"

        self.torch = torch.cuda if cuda else torch
        self.cuda = cuda
        self.seg = pseg

        if model_source is not None:
            if self.cuda:
                model_source = torch.load(model_source)
            else:
                model_source = torch.load(
                    model_source, map_location=lambda storage, loc: storage)
            self.word2idx = model_source["word2idx"]
            self.char2idx = model_source["char2idx"]
            self.predicate2id = model_source["predicate2id"]
            self.args = args = model_source["settings"]
            self.max_len = model_source["max_len"]
            model = Model(args)
            model.load_state_dict(model_source['model'])

            if self.cuda:
                model = model.cuda()
            else:
                model = model.cpu()
            self.model = model
        else:
            self.model = model_datas["model"]
            self.word2idx = model_datas["word2idx"]
            self.char2idx = model_datas["char2idx"]
            self.predicate2id = model_datas["predicate2id"]
            self.max_len = model_datas["max_len"]

        self.model.eval()
        self.id2predicate = {v: k for k, v in self.predicate2id.items()}

    def update_model(self, model):
        self.model = model
        self.model.eval()

    def segment(self, text):
        words = []
        for w in [[e]*len(e) for e, _ in self.seg.cut(text)]:
            words += w

        return words

    def predict(self, text):
        triples = []
        if len(text) > self.max_len:
            text = text[:self.max_len]

        words = self.segment(text)
        c_id, w_id = common.q2idx(text, words, self.char2idx, self.word2idx)
        position = [pos_i+1 for pos_i in range(len(c_id))]

        with torch.no_grad():
            c_id, w_id, position = map(lambda x: self.torch.LongTensor(
                x).unsqueeze(0), (c_id, w_id, position))

            encode, shareFeat1, shareFeat2, attn_mask, _ = self.model.encode(
                c_id, w_id, position)
            sub_sidx, sub_eidx = self.model.sub_predict(
                encode, attn_mask, shareFeat1, shareFeat2)
            sub_sidx = sub_sidx.squeeze().gt(0.5).nonzero().tolist()
            sub_eidx = sub_eidx.squeeze().gt(0.4).nonzero().tolist()
            while len(sub_sidx) and len(sub_eidx):
                if sub_sidx[0][0] < sub_eidx[0][0]:
                    s, e = sub_sidx.pop(0)[0], sub_eidx.pop(0)[0]
                    subject = text[s:e+1]
                    sub_slidx = self.torch.LongTensor([[s]])
                    sub_elidx = self.torch.LongTensor([[e]])
                    obj_sidx, obj_eidx = self.model.obj_predict(
                        encode, shareFeat1, shareFeat2, sub_slidx, sub_elidx, attn_mask)

                    obj_sidxs = obj_sidx.squeeze().t()
                    obj_eidxs = obj_eidx.squeeze().t()
                    for idx, (sid, eid) in enumerate(zip(obj_sidxs, obj_eidxs)):
                        sids = sid.squeeze().gt(0.5).nonzero().tolist()
                        eids = eid.squeeze().gt(0.4).nonzero().tolist()
                        while len(sids) and len(eids):
                            if sids[0][0] < eids[0][0]:
                                s, e = sids.pop(0)[0], eids.pop(0)[0]
                                triples.append(
                                    (subject, self.id2predicate[idx], text[s:e+1]))
                                if self.id2predicate[idx] == "妻子":
                                    triples.append(
                                        (text[s:e+1], "丈夫", subject))
                                elif self.id2predicate[idx] == "丈夫":
                                    triples.append(
                                        (text[s:e+1], "妻子", subject))
                            else:
                                eids.pop(0)
                else:
                    sub_eidx.pop(0)

        return set(triples)


if __name__ == "__main__":
    import json
    import const
    predict = Predict(model_source="./weights/model_2.pt")
    triples = predict.predict("《李白》是李荣浩作词作曲并演唱的歌曲，该曲收录于2013年9月17号发行的原创专辑《模特》中")
    print(triples)
