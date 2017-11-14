import torch.nn as nn
from torch.autograd import Variable
import torch
import time
from torch.nn.functional import cosine_similarity

from module_utils import *

class FullMatchLay(nn.Module):
    def __init__(self, mp_dim, cont_dim):
        super().__init__()
        self.cont_dim = cont_dim
        self.mp_dim = mp_dim

        self.register_parameter("weight", nn.Parameter(torch.Tensor(mp_dim, cont_dim)))
        self.weight.data.uniform_(-1., 1.)

    def forward(self, cont_repres, other_cont_first):
        """
        Args:
            cont_repres - [batch_size, this_len, context_lstm_dim]
            other_cont_first - [batch_size, context_lstm_dim]
        Return:
            size - [batch_size, this_len, mp_dim]
        """
        def expand(context, weight):
            """
            Args:
                [batch_size, this_len, context_lstm_dim]
                [mp_dim, context_lstm_dim]
            Return:
                [batch_size, this_len, mp_dim, context_lstm_dim]
            """
            # [1, 1, mp_dim, context_lstm_dim]
            weight = weight.unsqueeze(0)
            weight = weight.unsqueeze(0)
            # [batch_size, this_len, 1, context_lstm_dim]
            context = context.unsqueeze(2)
            return torch.mul(context, weight)

        cont_repres = expand(cont_repres, self.weight)

        other_cont_first = multi_perspective_expand_for_2D(other_cont_first, self.weight)
        # [batch_size, 1, mp_dim, context_lstm_dim]
        other_cont_first = other_cont_first.unsqueeze(1)
        return cosine_similarity(cont_repres, other_cont_first, cont_repres.dim()-1)

class MaxpoolMatchLay(nn.Module):
    def __init__(self, mp_dim, cont_dim):
        super().__init__()
        self.cont_dim = cont_dim
        self.mp_dim = mp_dim

        self.register_parameter("weight", nn.Parameter(torch.Tensor(mp_dim, cont_dim)))
        self.weight.data.uniform_(-1., 1.)

    def forward(self, cont_repres, other_cont_repres):
        """
        Args:
            cont_repres - [batch_size, this_len, context_lstm_dim]
            other_cont_repres - [batch_size, other_len, context_lstm_dim]
        Return:
            size - [bsz, this_len, mp_dim*2]
        """
        bsz = cont_repres.size(0)
        this_len = cont_repres.size(1)
        other_len = other_cont_repres.size(1)

        cont_repres = cont_repres.view(-1, self.cont_dim)
        other_cont_repres = other_cont_repres.view(-1, self.cont_dim)

        cont_repres = multi_perspective_expand_for_2D(cont_repres, self.weight)
        other_cont_repres = multi_perspective_expand_for_2D(other_cont_repres, self.weight)

        cont_repres = cont_repres.view(bsz, this_len, self.mp_dim, self.cont_dim)
        other_cont_repres = other_cont_repres.view(bsz, other_len, self.mp_dim, self.cont_dim)

        # [bsz, this_len, 1, self.mp_dim, self.cont_dim]
        cont_repres = cont_repres.unsqueeze(2)
        # [bsz, 1, other_len, self.mp_dim, self.cont_dim]
        other_cont_repres = other_cont_repres.unsqueeze(1)

        # [bsz, this_len, other_len, self.mp_dim]fanruan
        simi = cosine_similarity(cont_repres, other_cont_repres, cont_repres.dim()-1)

        t_max, _ = simi.max(2)
        t_mean = simi.mean(2)
        return torch.cat((t_max, t_mean), 2)

class AtteMatchLay(nn.Module):
    def __init__(self, mp_dim, cont_dim):
        super(AtteMatchLay, self).__init__()
        self.cont_dim = cont_dim
        self.mp_dim = mp_dim

        self.register_parameter("weight", nn.Parameter(torch.Tensor(mp_dim, cont_dim)))
        self.weight.data.uniform_(-1., 1.)

    def forward(self, repres, max_att):
        """
        Args:
            repres - [bsz, a_len|q_len, cont_dim]
            max_att - [bsz, q_len|a_len, cont_dim]
        Return:
            size - [bsz, sentence_len, mp_dim]
        """
        bsz = repres.size(0)
        sent_len = repres.size(1)

        repres = repres.view(-1, self.cont_dim)
        max_att = max_att.view(-1, self.cont_dim)
        repres = multi_perspective_expand_for_2D(repres, self.weight)
        max_att = multi_perspective_expand_for_2D(max_att, self.weight)
        temp = cosine_similarity(repres, max_att, repres.dim()-1)

        return temp.view(bsz, sent_len, self.mp_dim)
