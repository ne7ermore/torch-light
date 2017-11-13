import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from const import *
from model import Transformer
from utils import normalizeString

import time
import copy

class Translate(object):
    def __init__(self, model_source, cuda=False, beam_size=3):
        self.torch = torch.cuda if cuda else torch
        self.cuda = cuda
        self.beam_size = beam_size

        if self.cuda:
            model_source = torch.load(model_source)
        else:
            model_source = torch.load(model_source, map_location=lambda storage, loc: storage)
        self.src_dict = model_source["src_dict"]
        self.tgt_dict = model_source["tgt_dict"]
        self.src_idx2word = {v: k for k, v in model_source["tgt_dict"].items()}
        self.args = args = model_source["settings"]
        model = Transformer(args)
        model.load_state_dict(model_source['model'])

        if self.cuda: model = model.cuda()
        else: model = model.cpu()
        self.model = model.eval()

    def sent2tenosr(self, sentence):
        max_len = self.args.max_word_len - 2
        sentence = normalizeString(sentence)
        words = [w for w in sentence.strip().split()]

        if len(words) > max_len:
            words = words[:max_len]

        words = [WORD[BOS]] + words + [WORD[EOS]]
        idx = [self.src_dict[w] if w in self.src_dict else UNK for w in words]

        idx_data = torch.LongTensor(idx)
        idx_position = torch.LongTensor([pos_i+1 if w_i != PAD else 0 for pos_i, w_i in enumerate(idx)])
        idx_data_tensor = Variable(idx_data.unsqueeze(0), volatile=True)
        idx_position_tensor = Variable(idx_position.unsqueeze(0), volatile=True)

        if self.cuda:
            idx_data_tensor = idx_data_tensor.cuda()
            idx_position_tensor = idx_position_tensor.cuda()

        return idx_data_tensor, idx_position_tensor

    def beam_search(self, w_scores, top_seqs):
        max_scores, max_idxs = w_scores.squeeze().sort(-1, descending=True)
        max_scores = (max_scores[:, :self.beam_size]).tolist()
        max_idxs = (max_idxs[:, :self.beam_size]).tolist()

        all_seqs = []

        for index, seq in enumerate(top_seqs):
            seq_idxs, seq_score = seq
            last_seq_idx = seq_idxs[-1]
            if last_seq_idx == EOS:
                all_seqs += [(seq, seq_score, True)]
                continue

            for score, idx in zip(max_scores[index], max_idxs[index]):
                temp_seq = copy.deepcopy(seq)
                seq_idxs, seq_score = temp_seq
                seq_score += score
                seq_idxs += [idx]
                all_seqs += [((seq_idxs, seq_score), seq_score, idx == EOS)]

        top_seqs = sorted(all_seqs, key = lambda seq: seq[1], reverse=True)[:self.beam_size]

        all_done, done_nums = self.check_all_done(top_seqs)
        top_seqs = [seq for seq, _, _ in top_seqs]

        return top_seqs, all_done, self.beam_size-done_nums

    def check_all_done(self, seqs):
        done_nums = len([s for s in seqs if s[-1]])
        return done_nums == self.beam_size, done_nums

    def init_input(self):
        input_data = torch.LongTensor(self.beam_size).fill_(BOS).unsqueeze(1)
        return Variable(input_data.long(), volatile=True)

    def update_input(self, top_seqs):
        input_data = [seq[0] for seq in top_seqs if seq[0][-1] != EOS]
        input_data = torch.LongTensor(input_data)
        return Variable(input_data, volatile=True)

    def update_state(self, step, src_seq, enc_outputs, un_dones):
        input_pos = torch.arange(1, step+1).unsqueeze(0)
        input_pos = input_pos.repeat(un_dones, 1)
        input_pos = Variable(input_pos.long(), volatile=True)

        src_seq_beam = Variable(src_seq.data.repeat(un_dones, 1))
        enc_outputs_beam = [Variable(enc_output.data.repeat(un_dones, 1, 1)) for enc_output in enc_outputs]

        return input_pos, src_seq_beam, enc_outputs_beam

    def decode(self, seq, pos):
        def length_penalty(step, len_penalty_w=1.):
            return (torch.log(self.torch.FloatTensor([5 + step])) - torch.log(self.torch.FloatTensor([6])))*len_penalty_w

        top_seqs = [([BOS], 0)] * self.beam_size

        enc_outputs = self.model.enc(seq, pos)
        seq_beam = Variable(seq.data.repeat(self.beam_size, 1))
        enc_outputs_beam = [Variable(enc_output.data.repeat(self.beam_size, 1, 1)) for enc_output in enc_outputs]

        input_data = self.init_input()
        input_pos = torch.arange(1, 2).unsqueeze(0)
        input_pos = input_pos.repeat(self.beam_size, 1)
        input_pos = Variable(input_pos.long(), volatile=True)

        for step in range(1, self.args.max_word_len+1):
            if self.cuda:
                input_pos = input_pos.cuda()
                input_data = input_data.cuda()

            dec_output = self.model.dec(enc_outputs_beam,
                            seq_beam, input_data, input_pos)
            dec_output = dec_output[:, -1, :] # word level feature
            out = F.log_softmax(self.model.linear(dec_output))
            lp = length_penalty(step)

            top_seqs, all_done, un_dones = self.beam_search(out.data+lp, top_seqs)

            if all_done: break
            input_data = self.update_input(top_seqs)
            input_pos, src_seq_beam, enc_outputs_beam = self.update_state(step+1, seq, enc_outputs, un_dones)

        tgts = []
        for seq in top_seqs:
            cor_idxs, score = seq
            cor_idxs = cor_idxs[1: -1]
            tgts += [(" ".join([self.src_idx2word[idx] for idx in cor_idxs]), score)]
        return tgts

    def Trains(self, sentence, topk=1):
        idx_data, idx_pos = self.sent2tenosr(sentence)
        answers = self.decode(idx_data, idx_pos)
        assert topk <= len(answers)
        return [ans[0] for ans in answers[:topk]]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Translate')
    parser.add_argument('--French', type=str, required=True,
                    help='French for translating to English')
    args = parser.parse_args()
    pre = Translate("seq2seq.pt")
    print("Translated - {}".format(pre.Trains(args.French)[0]))
