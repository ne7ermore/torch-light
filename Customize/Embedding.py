import torch
import torch.nn as nn

import numpy as np

# using pre-trained word2vec init embedding
class PreEmbedding(nn.Module):
    def __init__(self, pre_w2v, vocab_size, ebd_dim):
        self.lookup_table = nn.Embedding(vocab_size, ebd_dim)

        # load pre-trained word2vec
        # 1st: check data type
        assert isinstance(pre_w2v, np.ndarray)

        # 2st: load
        self.lookup_table.weight.data.copy_(torch.from_numpy(pre_w2v))

        # 3st: frozen embedding weight
        self.lookup_table.weight.requires_grad = False

    def forward(self, x):
        return self.lookup_table(x)
