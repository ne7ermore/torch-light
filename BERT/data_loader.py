import random

import numpy as np
import torch
from torch.utils.data import Dataset

from const import *


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class BERTDataSet(Dataset):
    def __init__(self, sents, max_len, word_dict, steps):
        self.sents_size = len(sents)
        self.max_len = max_len
        self.word_size = len(word_dict)
        self.sents = np.asarray(sents)
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, item):
        inp, sent_label, word_label, segment_label = self.gen_one()

        pos = torch.LongTensor(
            [pos + 1 if w != PAD else 0 for pos, w in enumerate(inp)])
        inp = torch.LongTensor(inp)
        sent_label = torch.LongTensor([sent_label])
        word_label = torch.LongTensor(word_label)
        segment_label = torch.LongTensor(segment_label)

        return inp, pos, sent_label, word_label, segment_label

    def random_sample(self):
        idx = np.random.choice(self.sents_size, 1)
        return self.sents[idx][0]

    def gather_sent(self):
        sent = self.random_sample()
        if random.random() > RANDOM_SENT:
            return sent[0], sent[1], NEXT
        else:
            return sent[0], self.random_sample()[1], NOT_NEXT

    def gather_word(self, sent):
        label = []
        for idx, word in enumerate(sent):
            if random.random() < RANDOM_WORD_SAMPLE:
                if random.random() < RANDOM_MARK:
                    sent[idx] = MASK
                else:
                    if random.random() < RANDOM_WORD:
                        sent[idx] = np.random.choice(self.word_size, 1)[0]
                label.append(word)
            else:
                label.append(PAD)

        return sent, label

    def pad_sent(self, sent):
        return sent + (self.max_len - len(sent)) * [PAD]

    def gen_one(self):
        t1, t2, label = self.gather_sent()
        truncate_seq_pair(t1, t2, self.max_len - 3)

        t1, t1_label = self.gather_word(t1)
        t2, t2_label = self.gather_word(t2)

        t1 = [CLS] + t1 + [SEP]
        t2 = t2 + [SEP]

        t1_label = [PAD] + t1_label + [PAD]
        t2_label = t2_label + [PAD]

        one = self.pad_sent(t1 + t2)
        word_label = self.pad_sent(t1_label + t2_label)
        segment_label = self.pad_sent([SEGMENTA for _ in range(
            len(t1))] + [SEGMENTB for _ in range(len(t2))])

        return one, label, word_label, segment_label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data = torch.load("data/corpus.pt")
    ds = BERTDataSet(data["word"], data["max_len"], data["dict"], 10000)
    train_data_loader = DataLoader(ds, batch_size=1, num_workers=1)
    for inp, pos, sent_label, word_label, segment_label in train_data_loader:
        print("=" * 90)
        print(inp.shape)
        print(pos.shape)
        print(sent_label.shape)
        print(word_label.shape)
        print(segment_label.shape)
        print("=" * 90)
        print(word_label.shape)
        print((word_label > 0).float().sum())

    itow = {v: k for k, v in data["dict"].items()}

    for _ in range(10):
        one, label, word_label, segment_label = ds.gen_one()
        print(label)
        print(" ".join([itow[w] for w in one]))
        print(" ".join([itow[w] for w in word_label]))
        print(segment_label)
        print("=" * 30)
