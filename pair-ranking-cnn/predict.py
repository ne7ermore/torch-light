import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import const
from module import CNN_Ranking
from utils import prepare, corpora2idx

class Predict(object):
    def __init__(self, model_source, cuda=False, beam_size=3):
        self.torch = torch.cuda if cuda else torch
        self.cuda = cuda
        self.jb = Jieba("./segmenter_dicts", useSynonym=True, HMM=False)
        self.swf = StopwordFilter("./segmenter_dicts/stopwords.txt")

        model_source = torch.load(model_source)
        self.src_dict = model_source["src_dict"]
        self.tgt_dict = model_source["tgt_dict"]
        self.src_idx2ind = {v: k for k, v in model_source["tgt_dict"].items()}
        self.args = args = model_source["settings"]
        model = CNN_Ranking(args)
        model.load_state_dict(model_source['model'])

        if self.cuda:
            model = model.cuda()
        else:
            model = model.cpu()
        self.model = model.eval()

    def sent2corpora(self, sentence, synonym=False):
        sentence = prepare(sentence)
        corpora = [e[0] for e in self.jb.segment(sentence) if self.swf.filter(e[0])]
        new_corpora = []
        for corpus in corpora:
            if synonym and corpus in self.jb.synonym:
                corpus = self.jb.synonym[corpus]
            new_corpora.append(corpus)
        return new_corpora

    def sent2tensor(self, question, answers):
        q_max_len = self.args.max_lenth_src
        a_max_len = self.args.max_lenth_tgt
        src_corpora = [word for word in question if self.swf.filter(word)]
        if len(src_corpora) > q_max_len:
            src_corpora = src_corpora[:q_max_len]
        else:
            src_corpora += [const.WORD[const.PAD]]*(a_max_len-len(src_corpora))
        src_corpora = [self.src_dict[corpus] for corpus in src_corpora]

        q_tensor = torch.LongTensor(src_corpora).unsqueeze(0)
        q_tensor = Variable(q_tensor.repeat(len(answers), 1), volatile=True)

        tgt_sents = []
        for answer in answers:
            tgt_corpora = self.sent2corpora(answer)
            if len(tgt_corpora) > a_max_len:
                tgt_corpora = tgt_corpora[:a_max_len]
            else:
                tgt_corpora += [const.WORD[const.PAD]]*(a_max_len-len(tgt_corpora))
            tgt_sents.append(tgt_corpora)

        a_tensor = corpora2idx(tgt_sents, self.tgt_dict)
        a_tensor = Variable(torch.LongTensor(a_tensor), volatile=True)

        if self.cuda:
            return q_tensor.cuda(), a_tensor.cuda()
        else:
            return q_tensor, a_tensor


    def process(self, q_tensor, a_tensor, answers, top_k):
        pred = self.model(q_tensor, a_tensor)
        scores, indexs = pred.sort(dim=0, descending=True)
        hit_scores, hit_indexs = scores.data.chunk(2, dim=1), indexs.data.chunk(2, dim=1)
        print(hit_scores)
        hit_scores = hit_scores[1].squeeze().tolist()
        hit_indexs = hit_indexs[1].squeeze().tolist()
        best_answers = []
        top_k = min(top_k, pred.size(0))
        for k_num, (score, index) in enumerate(zip(hit_scores, hit_indexs)):
            if (k_num-1) == top_k: break
            best_answers += [(answers[index], score)]

        return best_answers

    def predict(self, question, answers, top_k=3):
        q_tensor, a_tensor = self.sent2tensor(question, answers)
        answers = self.process(q_tensor, a_tensor, answers, top_k)
        return answers
