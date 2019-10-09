import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

import const


def get_padding_mask(x):
    return x.gt(0).unsqueeze(2).float()


def gather_index(encode, k1, k2, n=6):
    x = torch.arange(start=0, end=n/(n-1.), step=1./(n-1), dtype=torch.float)
    if k1.is_cuda:
        x = x.cuda()

    k1 = x*(k1.float())
    k2 = (1-x)*(k2.float())
    index = torch.round(k1+k2).long()
    return torch.stack([torch.index_select(encode[idx], 0, index[idx]) for idx in range(encode.size(0))], dim=0)


def get_attn_padding_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    bsz, len_q = seq_q.size()
    pad_attn_mask = seq_k.data.eq(const.PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(bsz, len_q, len_q)
    return pad_attn_mask


class DilatedGatedConv1D(nn.Module):
    def __init__(self, dilation_rate, dim):
        super().__init__()

        self.dim = dim
        self.dropout = nn.Dropout(p=0.1)
        self.cnn = nn.Conv1d(
            dim, dim*2, 3, padding=dilation_rate, dilation=dilation_rate)

    def forward(self, x):
        residual = x
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x1, x2 = x[:, :, :self.dim], x[:, :, self.dim:]
        x1 = torch.sigmoid(self.dropout(x1))
        return residual*(1-x1) + x2*x1


class DgCNN(nn.Module):
    def __init__(self, dim, dilation_rates: list):
        super().__init__()

        self.cnn1ds = nn.ModuleList(
            [DilatedGatedConv1D(dilation_rate, dim) for dilation_rate in dilation_rates])

    def forward(self, x, mask):
        for layer in self.cnn1ds:
            x = layer(x)*mask
        return x


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

    def forward(self, q, k, v, attn_mask):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn.view(-1, attn.size(2))).view(*attn.size())
        attn = self.dropout(attn)
        return torch.bmm(attn, v)


class MultiHeadAtt(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.5):
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

    def forward(self, q, k, v, attn_mask):
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

        outputs = self.attention(q_s, k_s, v_s, attn_mask.repeat(n_head, 1, 1))
        outputs = torch.cat(torch.split(outputs, bsz, dim=0),
                            dim=-1).view(-1, n_head * d_v)
        outputs = self.dropout(self.w_o(outputs)).view(bsz, len_q, -1)
        return self.lm(outputs + residual)


class SubjectLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)


class SubModel(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.cnn = nn.Conv1d(2*dim, dim, 3, padding=1)
        self.lr1 = nn.Linear(dim, 1)
        self.lr2 = nn.Linear(dim, 1)

    def forward(self, x):
        x = F.relu(self.cnn(x.transpose(1, 2))).transpose(1, 2)
        x1 = torch.sigmoid(self.lr1(x))
        x2 = torch.sigmoid(self.lr2(x))

        return x1, x2


class ObjModel(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.cnn = nn.Conv1d(4*dim, dim, 3, padding=1)
        self.lr1 = nn.Linear(dim, 1)
        self.lr2 = nn.Linear(dim, num_classes)
        self.lr3 = nn.Linear(dim, num_classes)

    def forward(self, x, shareFeat1, shareFeat2):
        x = F.relu(self.cnn(x.transpose(1, 2))).transpose(1, 2)
        x1 = torch.sigmoid(self.lr1(x))
        x2 = torch.sigmoid(self.lr2(x))
        x3 = torch.sigmoid(self.lr3(x))

        x2 = x2*shareFeat1*x1
        x3 = x3*shareFeat2*x1

        return x2, x3


class ObjectRnn(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.rnn = nn.GRU(d_model,
                          hidden_size=d_model,
                          batch_first=True,
                          bidirectional=True)
        self.ln = LayerNorm(d_model*2)

    def forward(self, x, sub_slidx, sub_elidx, pos_ebd):
        idx = gather_index(x, sub_slidx, sub_elidx)
        encode, _ = self.rnn(idx)
        encode = self.ln(encode)[:, -1, :].unsqueeze(1)

        pos_ebd = self.position(x, sub_slidx, sub_elidx, pos_ebd)

        return encode+pos_ebd

    def position(self, x, sidx, eidx, pos_ebd):
        bsz, length, _ = x.size()
        pos_idx = torch.arange(0, length).repeat(bsz, 1)
        if x.is_cuda:
            pos_idx = pos_idx.cuda()

        s_pos = pos_ebd(torch.abs(pos_idx-sidx.long()))
        e_pos = pos_ebd(torch.abs(pos_idx-eidx.long()))

        return torch.cat((s_pos, e_pos), dim=-1)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.char_ebd = nn.Embedding(args.char_size, args.inp_dim)
        self.word_ebd = nn.Embedding(args.word_size, args.inp_dim)
        self.pos_ebd = nn.Embedding(args.max_len+1, args.d_model)

        self.char_lr = nn.Linear(args.inp_dim, args.d_model)
        self.word_lr = nn.Linear(args.inp_dim, args.d_model)
        self.encode_dropout = nn.Dropout(0.25)

        self.dgcnn = DgCNN(args.d_model, args.dilation_rates)

        self.sl1 = SubjectLinear(args.d_model)
        self.sl2 = SubjectLinear(args.d_model)

        self.sbj_att = MultiHeadAtt(args.n_head, args.d_model)

        self.subModel = SubModel(args.d_model)

        self.objectRnn = ObjectRnn(args.d_model)
        self.obj_att = MultiHeadAtt(args.n_head, args.d_model)
        self.objModel = ObjModel(args.d_model, args.num_classes)

        self._reset_parameters()

        if "charW" in args.__dict__ and "wordW" in args.__dict__:
            self.use_vecs()
        else:
            self.char_ebd.weight.data.uniform_(-.1, .1)
            self.word_ebd.weight.data.uniform_(-.1, .1)

    def use_vecs(self):
        args = self.args

        self.char_ebd.weight.data.copy_(torch.from_numpy(args.charW))
        self.word_ebd.weight.data.copy_(torch.from_numpy(args.wordW))

    def _reset_parameters(self):
        self.pos_ebd.weight.data.uniform_(-.1, .1)
        for layer in self.modules():
            if type(layer) == nn.Linear:
                layer.weight.data.normal_(std=const.INIT_RANGE)

    def encode(self, chars, words, posits):
        mask = get_padding_mask(chars)
        attn_mask = get_attn_padding_mask(chars, chars)
        ebd_encode = self.char_lr(self.char_ebd(
            chars)) + self.word_lr(self.word_ebd(words)) + self.pos_ebd(posits)

        ebd_encode = self.encode_dropout(ebd_encode)
        encode = self.dgcnn(ebd_encode, mask)

        shareFeat1 = self.sl1(encode)
        shareFeat2 = self.sl2(encode)

        return encode, shareFeat1, shareFeat2, attn_mask, mask

    def sub_predict(self, encode, attn_mask, shareFeat1, shareFeat2):
        attn = self.sbj_att(encode, encode, encode, attn_mask)
        output = torch.cat((attn, encode), dim=-1)

        sub_sidx, sub_eidx = self.subModel(output)

        return sub_sidx*shareFeat1, sub_eidx*shareFeat2

    def obj_predict(self, encode, shareFeat1, shareFeat2, sub_slidx, sub_elidx, attn_mask):
        rnn_encode = self.objectRnn(encode, sub_slidx, sub_elidx, self.pos_ebd)

        attn = self.obj_att(encode, encode, encode, attn_mask)
        encode = torch.cat((attn, encode, rnn_encode), dim=-1)

        obj_sidx, obj_eidx = self.objModel(encode, shareFeat1, shareFeat2)

        return obj_sidx, obj_eidx

    def forward(self, chars, words, posits, sub_slidx, sub_elidx):
        encode, shareFeat1, shareFeat2, attn_mask, mask = self.encode(
            chars, words, posits)
        sub_sidx, sub_eidx = self.sub_predict(
            encode, attn_mask, shareFeat1, shareFeat2)
        obj_sidx, obj_eidx = self.obj_predict(
            encode, shareFeat1, shareFeat2, sub_slidx, sub_elidx, attn_mask)

        return sub_sidx, sub_eidx, obj_sidx, obj_eidx, mask

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
    from data_loader import DataLoader
    import argparse
    import const
    import os

    data = torch.load(os.path.join(const.DATAPATH, "corpus.pt"))
    dl = DataLoader(data["train"]["char"],
                    data["train"]["word"],
                    data["train"]["sub_sidx"],
                    data["train"]["sub_eidx"],
                    data["train"]["obj_sidx"],
                    data["train"]["obj_eidx"],
                    data["train"]["sub_slidx"],
                    data["train"]["sub_elidx"],
                    data["word2idx"],
                    data["char2idx"],
                    data["predicate2id"],
                    batch_size=32)

    chars, words, position, sub_sidx, sub_eidx, obj_sidx, obj_eidx, sub_slidx, sub_elidx = next(
        dl)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dilation_rates', type=str,
                        default='1,2,5,1,2,5,1,2,5,1,1,1')
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)

    args = parser.parse_args()
    args.max_len = data["max_len"]
    args.char_size = len(data["char2idx"])
    args.word_size = len(data["word2idx"])
    args.num_classes = len(data["predicate2id"])
    args.inp_dim = 200
    args.dilation_rates = list(map(int, args.dilation_rates.split(",")))

    model = Model(args)
    model = model.cuda()
    sub_sidx, sub_eidx, obj_sidx, obj_eidx, mask = model(
        chars, words, position, sub_slidx, sub_elidx)
    print(sub_sidx.shape, sub_eidx.shape, obj_sidx.shape, obj_eidx.shape)
    print(mask.tolist()[0])

    id2chars = {v: k for k, v in dl.char2idx.items()}
    print("".join([id2chars[idx] for idx in chars.tolist()[0]]))
