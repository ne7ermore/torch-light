import copy

import torch

from const import *
from model import Transformer
import common


class Predict(object):
    def __init__(self, model_source, rewrite_len=30, beam_size=4, debug=False):
        self.beam_size = beam_size
        self.rewrite_len = rewrite_len
        self.debug = debug

        model_source = torch.load(
            model_source, map_location=lambda storage, loc: storage)
        self.dict = model_source["word2idx"]
        self.idx2word = {v: k for k, v in model_source["word2idx"].items()}
        self.args = args = model_source["settings"]
        torch.manual_seed(args.seed)
        model = Transformer(args)
        model.load_state_dict(model_source['model'])
        self.model = model.eval()

    def sent2tenosr(self, sentences):
        assert isinstance(sentences, list) and len(sentences) == 3

        max_len = self.args.max_context_len
        query1, query2, query3 = sentences
        q1_words = common.split_char(query1)
        turn1 = [1]*(len(q1_words))
        q2_words = common.split_char(query2)
        turn2 = [2]*(len(q2_words)+1)
        q3_words = common.split_char(query3)
        turn3 = [3]*(len(q3_words))
        words = q1_words + q2_words + [WORD[EOS]] + q3_words
        turns = turn1 + turn2 + turn3

        if len(words) > max_len:
            words = words[:max_len]

        idx = [self.dict[w] if w in self.dict else UNK for w in words]

        inp = torch.LongTensor(idx).unsqueeze(0)
        position = torch.LongTensor(
            [pos_i+1 if w_i != PAD else 0 for pos_i, w_i in enumerate(idx)]).unsqueeze(0)
        turns = torch.LongTensor(turns).unsqueeze(0)

        self.word = words

        return inp, position, turns

    def widx2didx(self, widx):
        word = self.word[widx]
        return self.dict[word] if word in self.dict else UNK

    def beam_search(self, w_scores, end_seqs, top_seqs):
        max_scores, max_idxs = w_scores.sort(-1, descending=True)
        max_scores = (max_scores[:, :self.beam_size]).tolist()
        max_idxs = (max_idxs[:, :self.beam_size]).tolist()

        all_seqs, seen = [], []
        for index, seq in enumerate(top_seqs):
            seq_idxs, word_index, seq_score = seq
            if seq_idxs[-1] == EOS:
                all_seqs += [(seq, seq_score, True)]
                continue

            for score, widx in zip(max_scores[index], max_idxs[index]):
                idx = self.widx2didx(widx)
                seq_idxs, word_index, seq_score = copy.deepcopy(seq)
                seq_score += score
                seq_idxs += [idx]
                word_index += [widx]
                if word_index not in seen:
                    seen.append(word_index)
                    all_seqs += [((seq_idxs, word_index, seq_score),
                                  seq_score, idx == EOS)]

        all_seqs += [((seq[0], seq[1], seq[-1]), seq[-1], True)
                     for seq in end_seqs]
        top_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)[
            :self.beam_size]

        all_done, done_nums = self.check_all_done(top_seqs)
        top_seqs = [seq for seq, _, _ in top_seqs]

        return top_seqs, all_done, self.beam_size-done_nums

    def check_all_done(self, seqs):
        done_nums = len([s for s in seqs if s[-1]])
        return done_nums == self.beam_size, done_nums

    def init_input(self):
        return torch.LongTensor(self.beam_size).fill_(BOS).unsqueeze(1)

    def update_input(self, top_seqs):
        end_seqs, un_end_seqs, input_data = [], [], []
        for seq in top_seqs:
            if seq[0][-1] != EOS:
                un_end_seqs.append(seq)
                input_data.append(seq[0])
            else:
                end_seqs.append(seq)

        return torch.LongTensor(input_data), end_seqs, un_end_seqs

    def update_state(self, step, src_seq, enc_output, un_dones):
        input_pos = torch.arange(1, step+1).unsqueeze(0)
        input_pos = input_pos.repeat(un_dones, 1)
        input_pos = input_pos.long()

        src_seq_beam = src_seq.data.repeat(un_dones, 1)
        enc_output_beam = enc_output.data.repeat(un_dones, 1, 1)

        return input_pos, src_seq_beam, enc_output_beam

    def divine(self, sentences):
        def length_penalty(step, len_penalty_w=1.):
            return (torch.log(torch.FloatTensor([5 + step])) - torch.log(torch.FloatTensor([6])))*len_penalty_w

        with torch.no_grad():
            inp, position, turns = self.sent2tenosr(sentences)

            top_seqs = [([BOS], [], 0)] * self.beam_size
            enc_output = self.model.encode(inp, position, turns)
            inp_beam = inp.data.repeat(self.beam_size, 1)
            enc_output_beam = enc_output.data.repeat(self.beam_size, 1, 1)
            input_data = self.init_input()
            end_seqs = []
            for step in range(1, self.rewrite_len):
                dec_output = self.model.decode(
                    input_data, inp_beam, enc_output_beam)
                out = dec_output[:, -1, :]
                lp = length_penalty(step)
                top_seqs, all_done, un_dones = self.beam_search(
                    out.data+lp, end_seqs, top_seqs)
                if all_done:
                    break

                input_data, end_seqs, top_seqs = self.update_input(top_seqs)
                input_pos, src_seq_beam, enc_output_beam = self.update_state(
                    step+1, inp, enc_output, un_dones)
                inp_beam = inp.data.repeat(un_dones, 1)

            tgts = []
            for (cor_idxs, word_index, score) in top_seqs:
                cor_idxs = word_index[: -1]
                tgts += [("".join([self.word[idx]
                                   for idx in cor_idxs]), score)]
            return tgts

    def Trains(self, sentences, topk=4):
        answers = self.divine(sentences)
        assert topk <= len(answers)
        if self.debug:
            print(answers)
        return [ans[0] for ans in answers[:topk]]


if __name__ == "__main__":
    pre = Predict("model.pt", debug=True)
    t1 = "你看莎士比亚吗"
    t2 = "最喜欢罗密欧与朱丽叶"
    t3 = "最喜欢那个角色"

    print(f"{t1}, {t2}, {t3} - {pre.Trains([t1,t2,t3])[0]}")
