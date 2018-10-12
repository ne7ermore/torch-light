import numpy as np
import pickle
import torch


class MiddleDataHandler(object):
    """
    save and load data for train
    """

    def __init__(self):
        self.pk = pickle

    def save(self, obj, _file):
        """
        args:
            obj: dict
            _file: the file to save obj
        """
        self.pk.dump(obj, open(_file, "wb"), True)

    def load(self, _file):
        """
        args:
            _file: load dict from the file

        return:
            obj
        """
        return self.pk.load(open(_file, "rb"))


def load_pre_w2c(_file, _dict):
    """
    load pre-train word2vec form file
    args:
        _file: word2vec file
        _dict: dictionary: map word to index, middle data which is for train

    return:
        word map from obj to vec
        type: matrix, numpy array
    """

    # contain w2v pre-train data
    # key: word
    # value: vec
    w2c_dict = {}

    # load and check length of vec
    for line in open(_file):
        temp = line.strip().split(" ")

        # discard first line
        if len(temp) < 10:
            continue
        w2c_dict[temp[0]] = list(map(float, temp[1:]))

        # length of vec, just one time
        if "len_" not in locals():
            len_ = len(temp[1:])

    # random init embedding: (0, 1]
    emb_mx = np.random.rand(len(_dict), len_)
    for word, idx in sorted(_dict.items(), key=lambda x: x[1]):
        if word in w2c_dict:
            emb_mx[idx] = np.asarray(w2c_dict[word])

    return emb_mx


def to_one_hot(y, n_dims=None):
    y_tensor = torch.tensor(y, dtype=torch.int64).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return y_one_hot
