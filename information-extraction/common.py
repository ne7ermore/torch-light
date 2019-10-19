import jieba.posseg as pseg
import numpy as np

import const


def q2idx(chars, words, char2idx, word2idx):
    chars = [char2idx[c] if c in char2idx else const.UNK for c in chars]
    words = [word2idx[w] if w in word2idx else const.UNK for w in words]

    return chars, words


def question2idx(questions, word2idx):
    return [[word2idx[w] if w in word2idx else const.UNK for w in question] for question in questions]


def load_pre_w2c(_file, char2idx, word2idx):
    w2c_dict = {}
    print("loading word2vec")
    for line in open(_file):
        temp = line.strip().split(" ")

        if len(temp) < 10:
            continue
        w2c_dict[temp[0]] = list(map(float, temp[1:]))

        if "len_" not in locals():
            len_ = len(temp[1:])

    print(f"load {len(w2c_dict)} lines word2vec")

    charW = np.random.rand(len(char2idx), len_)
    for word, idx in sorted(char2idx.items(), key=lambda x: x[1]):
        if word in w2c_dict:
            charW[idx] = np.asarray(w2c_dict[word])

    wordW = np.random.rand(len(word2idx), len_)
    for word, idx in sorted(word2idx.items(), key=lambda x: x[1]):
        if word in w2c_dict:
            wordW[idx] = np.asarray(w2c_dict[word])

    del w2c_dict
    return charW, wordW


class ScheduledOptim(object):
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        self.n_current_steps += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([np.power(
            self.n_current_steps, -0.5), np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class Jieba(object):
    def __init__(self):
        self.pseg = pseg

    def segment(self, text):
        words = self.pseg.cut(text)
        return [w for w, _ in words]


if __name__ == '__main__':
    jb = Jieba()
    print([e for e in jb.segment("《健行天下：带上一本健康的书去出行》一书的出版社是人民军医出版社，作者是秦惠基，出版时间是")])
