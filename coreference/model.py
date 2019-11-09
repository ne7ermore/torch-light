import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

import const


class Score(nn.Module):
    def __init__(self, in_dim, hidden_dim=150):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(in_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.score(x)


class GELU(nn.Module):
    """
    different from 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.gamma.expand_as(output) + self.beta.expand_as(output)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout):
        super().__init__()
        self.temper = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        attn = self.softmax(attn.view(-1, attn.size(2))).view(*attn.size())
        attn = self.dropout(attn)
        return torch.bmm(attn, v)


class MultiHeadAtt(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_v = self.d_k = d_k = d_model // n_head

        for name in ["w_qs", "w_ks", "w_vs"]:
            self.__setattr__(name,
                             nn.Parameter(torch.FloatTensor(n_head, d_model, d_k)))

        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.lm = LayerNorm(d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.w_qs.data.normal_(std=const.INIT_RANGE)
        self.w_ks.data.normal_(std=const.INIT_RANGE)
        self.w_vs.data.normal_(std=const.INIT_RANGE)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual = q

        bsz, len_q, d_model = q.size()
        len_k, len_v = k.size(1), v.size(1)

        def reshape(x):
            """[bsz, len, d_*] -> [n_head x (bsz*len) x d_*]"""
            return x.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        q_s, k_s, v_s = map(reshape, [q, k, v])

        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)

        outputs = self.attention(q_s, k_s, v_s)
        outputs = torch.cat(torch.split(outputs, bsz, dim=0),
                            dim=-1).view(-1, n_head * d_v)
        outputs = self.dropout(self.w_o(outputs)).view(bsz, len_q, -1)
        return self.lm(outputs + residual)


class Distance(nn.Module):

    bins = [1, 2, 3, 4, 8, 16, 32, 64]

    def __init__(self, distance_dim=20):
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.20)
        )

    def forward(self, lengths):
        return self.embeds(lengths)


class RnnEncoder(nn.Module):
    def __init__(self, d_model, embedding_dim, dropout):
        super().__init__()

        self.rnn = nn.GRU(embedding_dim,
                          hidden_size=d_model,
                          batch_first=True,
                          bidirectional=True)
        self.ln = LayerNorm(d_model*2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        encode, _ = self.rnn(x)
        encode = self.ln(encode)
        return self.dropout(encode)[:, -1, :]


class MentionPairScore(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.position_embedding = nn.Embedding(args.max_len+1, args.pos_dim)
        self.word_embedding = nn.Embedding(args.word_ebd_weight.shape[0],
                                           args.word_ebd_weight.shape[1])
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(args.word_ebd_weight))
        self.embedding_transform = nn.Linear(
            args.pos_dim+args.word_ebd_weight.shape[1], args.d_model)
        self.transform_activate = GELU()

        self.rnn_rncoder = RnnEncoder(args.rnn_hidden_size,
                                      args.word_ebd_weight.shape[1], args.dropout)

        self.dropout = nn.Dropout(args.dropout)

        self.head_att = MultiHeadAtt(args.n_head, args.d_model, args.dropout)
        self.distance_embedding = Distance()

        score_in_dim = 4*args.d_model + 4*args.rnn_hidden_size + args.pos_dim
        self.score = Score(score_in_dim)

        self._reset_parameters()

    def forward(self, doc, word2idx):
        doc_encoding = self.doc_encode(doc)
        mention_rnn_encode, coref_rnn_encode, distances_embedding_encode, corefs_idxs, mention_idxs, labels = self.mention_encode(
            doc, word2idx)
        doc_features = torch.stack(([torch.cat((doc_encoding[mention_start], doc_encoding[mention_end], doc_encoding[coref_start],
                                                doc_encoding[coref_end])) for (mention_start, mention_end), (coref_start, coref_end) in zip(mention_idxs, corefs_idxs)]), dim=0)

        mention_features = torch.cat(
            (doc_features, mention_rnn_encode, coref_rnn_encode, distances_embedding_encode), dim=1)
        scores = self.score(mention_features).squeeze()

        return torch.sigmoid(scores), labels

    def doc_encode(self, doc):
        doc_embedding_encode = self.word_embedding(doc.token_tensors)
        doc_postion_encode = self.position_embedding(
            doc.pos2tensor(self.args.use_cuda))
        doc_encode = self.dropout(torch.cat((
            doc_embedding_encode, doc_postion_encode), dim=-1))
        transform_encode = self.embedding_transform(doc_encode)
        transform_encode = self.transform_activate(
            transform_encode).unsqueeze(0)
        return self.head_att(transform_encode, transform_encode, transform_encode).squeeze()

    def mention_encode(self, doc, word2idx):
        corefs_idxs, mention_idxs, mention_spans, labels, distances, corefs = doc.sample(
            self.args.use_cuda, self.args.batch_size)
        distances_embedding_encode = self.distance_embedding(distances)
        mention_embedding_encode = self.word_embedding(mention_spans)
        coref_embedding_encode = self.word_embedding(corefs)

        distances_embedding_encode = self.dropout(distances_embedding_encode)
        mention_embedding_encode = self.dropout(mention_embedding_encode)
        coref_embedding_encode = self.dropout(coref_embedding_encode)

        mention_rnn_encode = self.rnn_rncoder(mention_embedding_encode)
        coref_rnn_encode = self.rnn_rncoder(coref_embedding_encode)

        return mention_rnn_encode, coref_rnn_encode, distances_embedding_encode, corefs_idxs, mention_idxs, labels

    def mention_predict(self, tokens, positions, mention, coref_idx, mention_idx, distance, coref):
        doc_embedding_encode = self.word_embedding(tokens)
        doc_postion_encode = self.position_embedding(positions)
        doc_encode = self.dropout(torch.cat((
            doc_embedding_encode, doc_postion_encode), dim=-1))
        transform_encode = self.embedding_transform(doc_encode)
        transform_encode = self.transform_activate(
            transform_encode).unsqueeze(0)
        doc_encoding = self.head_att(
            transform_encode, transform_encode, transform_encode).squeeze()

        distance_embedding_encode = self.distance_embedding(distance)
        mention_embedding_encode = self.word_embedding(mention)
        coref_embedding_encode = self.word_embedding(coref)

        distance_embedding_encode = self.dropout(distance_embedding_encode)
        mention_embedding_encode = self.dropout(mention_embedding_encode)
        coref_embedding_encode = self.dropout(coref_embedding_encode)

        mention_rnn_encode = self.rnn_rncoder(mention_embedding_encode)
        coref_rnn_encode = self.rnn_rncoder(coref_embedding_encode)

        doc_feature = torch.cat((doc_encoding[mention_idx[0]], doc_encoding[mention_idx[1]],
                                 doc_encoding[coref_idx[0]], doc_encoding[coref_idx[1]]))

        mention_feature = torch.cat((doc_feature.unsqueeze(
            0), mention_rnn_encode, coref_rnn_encode, distance_embedding_encode.squeeze(0)), dim=1)
        score = self.score(mention_feature).squeeze()
        return torch.sigmoid(score)

    def _reset_parameters(self):
        self.position_embedding.weight.data.uniform_(-.1, .1)
        for layer in self.modules():
            if type(layer) == nn.Linear:
                layer.weight.data.normal_(std=const.INIT_RANGE)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, cuda):
        if cuda:
            self.load_state_dict(torch.load(path))
            self.cuda()
        else:
            self.load_state_dict(torch.load(
                path, map_location=lambda storage, loc: storage))
            self.cpu()


if __name__ == "__main__":
    import data_loader
    import os
    import utils
    import argparse

    use_cuda = torch.cuda.is_available()
    corpus = torch.load(os.path.join(const.DATAPATH, "corpus.pt"))
    dl = data_loader.DataLoader(const.DATAPATH, corpus["word2idx"], cuda=False)
    doc = dl.sample_data()[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_len', type=int, default=500)
    parser.add_argument('--span_len', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--pos_dim', type=int, default=20)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--rnn_hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()

    args.word_ebd_weight = corpus["wordW"]
    args.use_cuda = use_cuda

    mps = MentionPairScore(args)

    scores, labels = mps(doc, corpus["word2idx"])
    print(scores.gt(0.5).long())
    print(scores.gt(0.5)*labels)
    print(labels.shape)
