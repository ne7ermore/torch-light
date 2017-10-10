import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import const
from modules import Transformer
from common.segmenter import Jieba
from common.word_filter import StopwordFilter
from utils import Trie, corpora2idx

class Predict(object):
    def __init__(self, model_source, cuda=False, beam_size=3):
        self.torch = torch.cuda if cuda else torch
        self.cuda = cuda
        self.beam_size = beam_size
        self.jb = Jieba("./segmenter_dicts", useSynonym=True, HMM=False)
        self.swf = StopwordFilter("./segmenter_dicts/stopwords.txt")

        model_source = torch.load(model_source)
        self.src_dict = model_source["src_dict"]
        self.tgt_dict = model_source["tgt_dict"]
        self.src_idx2ind = {v: k for k, v in model_source["tgt_dict"].items()}
        self.args = args = model_source["settings"]
        model = Transformer(
                args.src_vocab_size,
                args.tgt_vocab_size,
                args.max_token_seq_len,
                proj_share_weight=args.proj_share_weight,
                embs_share_weight=args.embs_share_weight,
                d_model=args.d_model,
                emb_dim=args.d_model,
                d_inner_hid=args.d_inner_hid,
                n_layers=args.n_layers,
                n_head=args.n_head,
                dropout=args.dropout)
        model.load_state_dict(model_source['model'])

        prob_projection = nn.LogSoftmax()
        if self.cuda:
            model = model.cuda()
            model.prob_projection = prob_projection.cuda()
        else:
            model = model.cpu()
            model.prob_projection = prob_projection.cpu()

        self.model = model.eval()

    def sent2idx(self, question, answers):
        max_len = self.args.max_token_seq_len - 2

        if self.args.by_word:
            src_corpora = [word for word in question if self.swf.filter(word)]
        else:
            src_corpora = self.sent2corpora(question)
        if len(src_corpora) > max_len:
            src_corpora = src_corpora[:max_len]

        src_sent = [const.WORD[const.BOS]] + src_corpora + [const.WORD[const.EOS]]
        src_sent = [self.src_dict[w] if w in self.src_dict else const.UNK for w in src_sent]

        tgt_sents = []
        for answer in answers:
            tgt_corpora = self.sent2corpora(answer)
            if len(tgt_corpora) > max_len:
                tgt_corpora = tgt_corpora[:max_len]
            tgt_sents.append([const.WORD[const.BOS]] + tgt_corpora + [const.WORD[const.EOS]])

        return src_sent, corpora2idx(tgt_sents, self.tgt_dict)

    def idx2tensor(self, sent_idxs):
        idx_data = torch.LongTensor(sent_idxs)
        idx_position = torch.LongTensor([pos_i+1 if w_i != const.PAD else 0 for pos_i, w_i in enumerate(idx_data)])
        idx_data_tensor = Variable(idx_data.unsqueeze(0), volatile=True)
        idx_position_tensor = Variable(idx_position.unsqueeze(0), volatile=True)

        if self.cuda:
            idx_data_tensor = idx_data_tensor.cuda()
            idx_position_tensor = idx_position_tensor.cuda()

        return (idx_data_tensor, idx_position_tensor)

    def sent2corpora(self, sentence, synonym=False):
        sentence = prepare(sentence)
        corpora = [e[0] for e in self.jb.segment(sentence) if self.swf.filter(e[0])]
        new_corpora = []
        for corpus in corpora:
            if synonym and corpus in self.jb.synonym:
                corpus = self.jb.synonym[corpus]
            new_corpora.append(corpus)
        return new_corpora

    def beam_search(self, vacab_scores, top_seqs):
        all_seqs = []
        index = 0
        for seq in top_seqs:
            seq_idxs, seq_score, answer_tree = seq
            node_index = seq_idxs[-1]
            if node_index == const.EOS:
                all_seqs += [(seq, seq_score, True)]
                continue

            this_v_scores = vacab_scores[index]
            idxs = list(answer_tree.keys())
            idx_tensor = self.torch.LongTensor(idxs)

            step_scores = this_v_scores.index_select(0, idx_tensor).tolist()
            corpora = zip(step_scores, idxs)
            corpora = sorted(corpora, key=lambda x: x[0], reverse=True)
            if len(corpora) > self.beam_size:
                corpora = corpora[:self.beam_size]

            index += 1
            for score, idx in corpora:
                new_tree = answer_tree[idx]
                this_score = score + seq_score
                this_seq = (seq_idxs + [idx], this_score, new_tree)
                all_seqs += [(this_seq, this_score, (idx == const.EOS))]

        top_seqs = sorted(all_seqs, key = lambda seq: seq[1], reverse=True)[:self.beam_size]
        all_done, done_nums = self.check_all_done(top_seqs)
        top_seqs = [seq for seq, _, _ in top_seqs]
        return top_seqs, all_done, self.beam_size-done_nums

    def check_all_done(self, seqs):
        done_nums = len([s for s in seqs if s[-1]])
        return done_nums == self.beam_size, done_nums

    def init_input(self):
        input_data = torch.LongTensor(self.beam_size).fill_(const.BOS).unsqueeze(1)
        return Variable(input_data.type(torch.LongTensor), volatile=True)

    def update_input(self, top_seqs):
        input_data = [seq[0] for seq in top_seqs if seq[0][-1] != const.EOS]
        input_data = torch.LongTensor(input_data)
        return Variable(input_data, volatile=True)

    def update_state(self, step, src_seq, enc_outputs, un_dones):
        input_pos = torch.arange(1, step+1).unsqueeze(0)
        input_pos = input_pos.repeat(un_dones, 1)
        input_pos = Variable(input_pos.type(torch.LongTensor), volatile=True)

        src_seq_beam = Variable(src_seq.data.repeat(un_dones, 1))
        enc_outputs_beam = [Variable(enc_output.data.repeat(un_dones, 1, 1)) for enc_output in enc_outputs]

        return input_pos, src_seq_beam, enc_outputs_beam

    def process(self, question, answers, len_penalty_w=0.3):
        def length_penalty(step, len_penalty_w):
            return (torch.log(self.torch.FloatTensor([5 + step])) - torch.log(self.torch.FloatTensor([6])))*len_penalty_w

        answer_tree = Trie()
        for answer in answers:
            answer_tree.add(answer[1:])

        top_seqs = [([const.BOS], 0, answer_tree.tree)]

        src_seq, src_pos = question
        enc_outputs, _ = self.model.encoder(src_seq, src_pos)
        src_seq_beam = Variable(src_seq.data.repeat(self.beam_size, 1))
        enc_outputs_beam = [Variable(enc_output.data.repeat(self.beam_size, 1, 1)) for enc_output in enc_outputs]

        input_data = self.init_input()
        input_pos = torch.arange(1, 2).unsqueeze(0)
        input_pos = input_pos.repeat(self.beam_size, 1)
        input_pos = Variable(input_pos.type(torch.LongTensor), volatile=True)

        for step in range(1, self.args.max_token_seq_len+1):
            if self.cuda:
                input_pos = input_pos.cuda()
                input_data = input_data.cuda()

            dec_outputs, _, _ = self.model.decoder(input_data, input_pos, src_seq_beam, enc_outputs_beam)

            dec_output = dec_outputs[-1][:, -1, :] # last one
            dec_output = self.model.tgt_word_proj(dec_output)
            out = self.model.prob_projection(dec_output)
            lp = length_penalty(step, len_penalty_w)

            top_seqs, all_done, un_dones = self.beam_search(out.data+lp, top_seqs)
            if all_done: break
            input_data = self.update_input(top_seqs)
            input_pos, src_seq_beam, enc_outputs_beam = self.update_state(step+1, src_seq, enc_outputs, un_dones)

        tgts = []
        for seq in top_seqs:
            cor_idxs, score, _ = seq
            cor_idxs = cor_idxs[1: -1] # <s></s>
            tgts += [("".join([self.src_idx2ind[idx] for idx in cor_idxs]), score)]
        return tgts

    def predict(self, question, answers):
        src_sent, tgt_sents = self.sent2idx(question, answers)
        src = self.idx2tensor(src_sent)
        answers = self.process(src, tgt_sents)
        return answers

if __name__ == "__main__":
    pre = Predict("model/attn_model")
    print(pre.predict("16寸的可以挂不麻烦您对照一下图纸，告知我这些板子的编号的呢，谢谢麻烦您对照一下图纸，告知我这些板子的编号的呢，谢谢麻烦您对照一下图纸，告知我这些板子的编号的呢，谢谢", ["亲亲给我订单号亲", "是不是填过订单", "麻烦您对照一下图纸，告知我这些板子的编号的呢，谢谢", "亲亲麻烦您整体拍照看看的", "订单号给我下", "亲714寸是摆台1648寸是挂画哦"]))
