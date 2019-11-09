import random
import os

import numpy as np
import torch

import const
import utils


class DataLoader:
    def __init__(self, inp_data, word2idx, cuda=True):
        self.cuda = cuda
        self.inp_data = inp_data
        self.word2idx = word2idx

        self.train_docs = self.load_files("train")
        self.test_docs = self.load_files("development")

        self.documents2tensor()

    def load_files(self, dtype):
        documents = []
        data_path = f"{self.inp_data}/data/{dtype}"
        for _, _, files in os.walk(data_path):
            for inf in files:
                if inf not in const.FILTERFILES and inf.endswith("conll"):
                    documents += utils.load_file(f"{data_path}/{inf}")

        return documents

    def documents2tensor(self):
        for doc in self.train_docs:
            doc.tokens2tensor(self.cuda, self.word2idx)
            doc.mentions(self.word2idx)
            doc.span2tonsor(self.word2idx)

        for doc in self.test_docs:
            doc.tokens2tensor(self.cuda, self.word2idx)
            doc.mentions(self.word2idx)
            doc.span2tonsor(self.word2idx)


if __name__ == "__main__":
    corpus = torch.load(os.path.join(const.DATAPATH, "corpus.pt"))
    dl = DataLoader(const.DATAPATH, corpus["word2idx"], cuda=False)

    # doc = dl.sample_data()[0]
    # corefs_idxs, mention_idxs, mention_spans, labels, distances, corefs = doc.sample(False, 20)
    # print(corefs_idxs, mention_idxs)
    for doc in dl.test_docs:
        if doc.mention_spans.shape[0] == 0:
            print(doc.filename.split("/")[-1])
