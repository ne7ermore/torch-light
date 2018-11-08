import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from const import *


def position(n_position, d_model):
    position_enc = np.array([[pos / np.power(10000, 2 * i / d_model)
                              for i in range(d_model)] for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.sin(position_enc[1:, 1::2])

    return torch.from_numpy(position_enc).float()


def get_attn_padding_mask(seq_q):
    assert seq_q.dim() == 2
    bsz, len_q = seq_q.size()
    pad_attn_mask = seq_q.data.eq(PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(bsz, len_q, len_q)
    return pad_attn_mask


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


class GELU(nn.Module):
    """
    different from 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWise(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1),
            GELU(),
            nn.Conv1d(d_ff, d_model, 1),
            nn.Dropout(dropout)
        )
        self.lm = LayerNorm(d_model)

    def forward(self, input):
        residual = input
        out = self.seq(input.transpose(1, 2)).transpose(1, 2)
        return self.lm(residual + out)


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
        self.dropout = dropout

        self.reset_parameters()

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
        outputs = F.dropout(self.w_o(outputs),
                            p=self.dropout).view(bsz, len_q, -1)
        return self.lm(outputs + residual)

    def reset_parameters(self):
        self.w_qs.data.normal_(INIT_RANGE)
        self.w_ks.data.normal_(INIT_RANGE)
        self.w_vs.data.normal_(INIT_RANGE)
        self.w_o.weight.data.normal_(INIT_RANGE)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super().__init__()
        self.mh = MultiHeadAtt(n_head, d_model, dropout)
        self.pw = PositionWise(d_model, d_ff, dropout)

    def forward(self, enc_input, slf_attn_mask):
        enc_output = self.mh(
            enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output = self.pw(enc_output)
        return enc_output


class Pooler(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.linear = nn.Linear(d_model, d_model)
        self.linear.weight.data.normal_(INIT_RANGE)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(x[:, 0])
        return F.tanh(x)


class BERT(nn.Module):
    def __init__(self, vsz, max_len, n_enc, d_model, d_ff, n_head, dropout):
        super().__init__()

        n_position = 2 * max_len + 4

        self.enc_ebd = nn.Embedding(vsz, d_model, padding_idx=PAD)
        self.seg_ebd = nn.Embedding(3, d_model, padding_idx=PAD)
        self.pos_ebd = nn.Embedding(n_position, d_model, padding_idx=PAD)
        self.pos_ebd.weight.data = position(n_position, d_model)
        self.pos_ebd.weight.requires_grad = False

        self.dropout = nn.Dropout(p=dropout)
        self.ebd_normal = LayerNorm(d_model)
        self.out_normal = LayerNorm(d_model)

        self.encodes = nn.ModuleList([
            EncoderLayer(d_model, d_ff, n_head, dropout) for _ in range(n_enc)])

        self.pooler = Pooler(d_model)
        self.transform = nn.Linear(d_model, d_model)  # word hidden layer
        self.gelu = GELU()

        self.sent_predict = nn.Linear(d_model, 2)
        self.word_predict = nn.Linear(d_model, vsz)

        self.reset_parameters()

    def reset_parameters(self):
        self.enc_ebd.weight.data.normal_(INIT_RANGE)
        self.seg_ebd.weight.data.normal_(INIT_RANGE)

        self.sent_predict.weight.data.normal_(INIT_RANGE)
        self.sent_predict.bias.data.zero_()

        self.transform.weight.data.normal_(INIT_RANGE)
        self.transform.bias.data.zero_()

        self.word_predict.weight = self.enc_ebd.weight  # share weights

    def forward(self, inp, pos, segment_label):
        encode = self.enc_ebd(
            inp) + self.seg_ebd(segment_label) + self.pos_ebd(pos)

        encode = self.dropout(self.ebd_normal(encode))

        slf_attn_mask = get_attn_padding_mask(inp)

        for layer in self.encodes:
            encode = layer(encode, slf_attn_mask)

        sent = F.log_softmax(self.sent_predict(self.pooler(encode)), dim=-1)

        word_enc = self.out_normal(self.gelu(self.transform(encode)))
        word = F.log_softmax(self.word_predict(word_enc), dim=-1)

        return word, sent

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())


class WordCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, props, tgt):
        tgt_props = props.gather(2, tgt.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        tgt_sum = mask.sum()
        loss = -(tgt_props * mask).sum() / tgt_sum

        props = F.softmax(props, dim=-1)
        _, index = torch.max(props, -1)
        corrects = ((index.data == tgt).float() * mask).sum()

        return loss, corrects, tgt_sum


class ScheduledOptim(object):
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        self.optimizer.step()
        self.update_learning_rate()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        self.n_current_steps += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([np.power(
            self.n_current_steps, -0.5), np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


if __name__ == "__main__":
    import data_loader
    from torch.utils.data import DataLoader

    data = torch.load("data/corpus.pt")
    ds = data_loader.BERTDataSet(
        data["word"], data["max_len"], data["dict"], 10000)
    train_data_loader = DataLoader(ds, batch_size=2, num_workers=5)
    s_criterion = torch.nn.CrossEntropyLoss()
    device_ids = [0, 2]
    b = BERT(ds.word_size, data["max_len"], 12, 768, 3072, 12, 0.1)
    b = b.cuda(device_ids[0])
    b = torch.nn.DataParallel(b, device_ids=device_ids)
    print(
        f"BERT have {sum(x.numel() for x in b.parameters())} paramerters in total")
    for datas in train_data_loader:
        inp, pos, sent_label, word_label, segment_label = list(
            map(lambda x: x.cuda(device_ids[0]), datas))

        word, sent = b(inp, pos, segment_label)
        print(word.shape)
        print(sent.shape)
        print(sent_label.shape)

        s_criterion(sent, sent_label.view(-1))
