import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

from const import *


def log_sum_exp(input, keepdim=False):
    assert input.dim() == 2
    max_scores, _ = input.max(dim=-1, keepdim=True)
    output = input - max_scores
    return max_scores + torch.log(torch.sum(torch.exp(output), dim=-1, keepdim=keepdim))


def gather_index(input, index):
    assert input.dim() == 2 and index.dim() == 1
    index = index.unsqueeze(1).expand_as(input)
    output = torch.gather(input, 1, index)
    return output[:, 0]


class CRF(nn.Module):
    def __init__(self, label_size, is_cuda):
        super().__init__()
        self.label_size = label_size
        self.transitions = nn.Parameter(
            torch.randn(label_size, label_size))
        self._init_weight()
        self.torch = torch.cuda if is_cuda else torch

    def _init_weight(self):
        init.xavier_uniform_(self.transitions)
        self.transitions.data[START, :].fill_(-10000.)
        self.transitions.data[:, STOP].fill_(-10000.)

    def _score_sentence(self, input, tags):
        bsz, sent_len, l_size = input.size()
        score = Variable(self.torch.FloatTensor(bsz).fill_(0.))
        s_score = Variable(self.torch.LongTensor([[START]] * bsz))

        tags = torch.cat([s_score, tags], dim=-1)
        input_t = input.transpose(0, 1)

        for i, words in enumerate(input_t):
            temp = self.transitions.index_select(1, tags[:, i])
            bsz_t = gather_index(temp.transpose(0, 1), tags[:, i + 1])
            w_step_score = gather_index(words, tags[:, i + 1])
            score = score + bsz_t + w_step_score

        temp = self.transitions.index_select(1, tags[:, -1])
        bsz_t = gather_index(temp.transpose(0, 1),
                             Variable(self.torch.LongTensor([STOP] * bsz)))
        return score + bsz_t

    def forward(self, input):
        bsz, sent_len, l_size = input.size()
        init_alphas = self.torch.FloatTensor(
            bsz, self.label_size).fill_(-10000.)
        init_alphas[:, START].fill_(0.)
        forward_var = Variable(init_alphas)

        input_t = input.transpose(0, 1)
        for words in input_t:
            alphas_t = []
            for next_tag in range(self.label_size):
                emit_score = words[:, next_tag].view(-1, 1)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var, True))
            forward_var = torch.cat(alphas_t, dim=-1)
        forward_var = forward_var + self.transitions[STOP].view(
            1, -1)
        return log_sum_exp(forward_var)

    def viterbi_decode(self, input):
        backpointers = []
        bsz, sent_len, l_size = input.size()

        init_vvars = self.torch.FloatTensor(
            bsz, self.label_size).fill_(-10000.)
        init_vvars[:, START].fill_(0.)
        forward_var = Variable(init_vvars)

        input_t = input.transpose(0, 1)
        for words in input_t:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.label_size):
                _trans = self.transitions[next_tag].view(
                    1, -1).expand_as(words)
                next_tag_var = forward_var + _trans
                best_tag_scores, best_tag_ids = torch.max(
                    next_tag_var, 1, keepdim=True)  # bsz
                bptrs_t.append(best_tag_ids)
                viterbivars_t.append(best_tag_scores)

            forward_var = torch.cat(viterbivars_t, -1) + words
            backpointers.append(torch.cat(bptrs_t, dim=-1))

        terminal_var = forward_var + self.transitions[STOP].view(1, -1)
        _, best_tag_ids = torch.max(terminal_var, 1)

        best_path = [best_tag_ids.view(-1, 1)]
        for bptrs_t in reversed(backpointers):
            best_tag_ids = gather_index(bptrs_t, best_tag_ids)
            best_path.append(best_tag_ids.contiguous().view(-1, 1))

        best_path.pop()
        best_path.reverse()

        return torch.cat(best_path, dim=-1)


class BiLSTM(nn.Module):
    def __init__(self, word_size, word_ebd_dim, kernel_num, lstm_hsz, lstm_layers, dropout, batch_size):
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hsz = lstm_hsz
        self.batch_size = batch_size

        self.word_ebd = nn.Embedding(word_size, word_ebd_dim)
        self.lstm = nn.LSTM(word_ebd_dim + kernel_num,
                            hidden_size=lstm_hsz // 2,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self._init_weights()

    def _init_weights(self, scope=1.):
        self.word_ebd.weight.data.uniform_(-scope, scope)

    def forward(self, words, char_feats, hidden=None):
        encode = self.word_ebd(words)
        encode = torch.cat((char_feats, encode), dim=-1)
        output, hidden = self.lstm(encode, hidden)
        return output, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.lstm_layers * 2, self.batch_size, self.lstm_hsz // 2).zero_()),
                Variable(weight.new(self.lstm_layers * 2, self.batch_size, self.lstm_hsz // 2).zero_()))


class CNN(nn.Module):
    def __init__(self, char_size, char_ebd_dim,
                 kernel_num, filter_size, dropout):
        super().__init__()

        self.char_size = char_size
        self.char_ebd_dim = char_ebd_dim
        self.kernel_num = kernel_num
        self.filter_size = filter_size
        self.dropout = dropout

        self.char_ebd = nn.Embedding(self.char_size, self.char_ebd_dim)
        self.char_cnn = nn.Conv2d(in_channels=1,
                                  out_channels=self.kernel_num,
                                  kernel_size=(self.filter_size, self.char_ebd_dim))
        self._init_weight()

    def _init_weight(self, scope=1.):
        init.xavier_uniform_(self.char_ebd.weight)

    def forward(self, input):
        bsz, word_len, char_len = input.size()
        encode = input.view(-1, char_len)
        encode = self.char_ebd(encode).unsqueeze(1)
        encode = F.relu(self.char_cnn(encode))
        encode = F.max_pool2d(encode,
                              kernel_size=(encode.size(2), 1))
        encode = F.dropout(encode.squeeze(), p=self.dropout)
        return encode.view(bsz, word_len, -1)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.cnn = CNN(self.char_size, self.char_ebd_dim,
                       self.kernel_num, self.filter_size, self.dropout)
        self.bilstm = BiLSTM(self.word_size, self.word_ebd_dim, self.kernel_num,
                             self.lstm_hsz, self.lstm_layers, self.dropout, self.batch_size)

        self.logistic = nn.Linear(self.lstm_hsz, self.label_size)
        self.crf = CRF(self.label_size, self.use_cuda)
        self._init_weights()

    def forward(self, words, chars, labels, hidden=None):
        char_feats = self.cnn(chars)
        output, _ = self.bilstm(words, char_feats, hidden)
        output = self.logistic(output)
        pre_score = self.crf(output)
        label_score = self.crf._score_sentence(output, labels)
        return (pre_score - label_score).mean(), None

    def predict(self, word, char):
        char_out = self.cnn(char)
        lstm_out, _ = self.bilstm(word, char_out)
        out = self.logistic(lstm_out)
        return self.crf.viterbi_decode(out)

    def _init_weights(self, scope=1.):
        self.logistic.weight.data.uniform_(-scope, scope)
        self.logistic.bias.data.fill_(0)
