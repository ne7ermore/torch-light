import torch
import argparse

import const
from common.word_filter import *
from common.segmenter import Jieba
import common.log as logging
from utils import corpora2idx

class Dictionary(object):
    def __init__(self):
        self.ind2idx = {
            const.WORD[const.BOS]: const.BOS,
            const.WORD[const.EOS]: const.EOS,
            const.WORD[const.PAD]: const.PAD,
            const.WORD[const.UNK]: const.UNK
        }
        self.idx2ind = {
            const.BOS: const.WORD[const.BOS],
            const.EOS: const.WORD[const.EOS],
            const.PAD: const.WORD[const.PAD],
            const.UNK: const.WORD[const.UNK]
        }
        self.idx = 4

    def add(self, ind):
        if self.ind2idx.get(ind) is None:
            self.ind2idx[ind] = self.idx
            self.idx2ind[self.idx] = ind
            self.idx += 1

    def build_idx(self, sents, min_count):
        corpora = [cor for sent in sents for cor in sent]
        word_count = {w: 0 for w in set(corpora)}
        for w in corpora: word_count[w]+=1

        ignored_word_count = 0
        for word, count in word_count.items():
            if count <= min_count:
                ignored_word_count += 1
                continue
            self.add(word)

        return ignored_word_count

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))

class Corpus(object):
    def __init__(self, train_src, save_data, valid_src=None,
                 max_len=32, min_word_count=5):
        self._train_src = train_src
        self._valid_src = valid_src
        self._save_data = save_data
        self._max_len = max_len
        self._min_word_count = min_word_count
        self.src_sents = None
        self.tgt_sents = None
        self.src_valid_sents = None
        self.tgt_valid_sents = None

        self.jb = Jieba("./segmenter_dicts", useSynonym=True, HMM=False) # unuse hmm
        self.swf = StopwordFilter("./segmenter_dicts/stopwords.txt")
        self.src_dict = Dictionary()
        self.tgt_dict = Dictionary()

    def sent2corpora(self, sentence, synonym=False):
        sentence = prepare(sentence)
        corpora = [e for e in self.jb.segment_search(sentence) if self.swf.filter(e)]
        new_corpora = []
        for corpus in corpora:
            if synonym and corpus in self.jb.synonym:
                corpus = self.jb.synonym[corpus]
            new_corpora.append(corpus)
        return new_corpora

    def parse_train(self, by_word=True):
        src_sents, tgt_sents = [], []
        for sentences in sopen(self._train_src):
            sentence = sentences.strip().split()
            if len(sentence) != 2: continue
            src_sent, tgt_sent = sentence
            max_len = self._max_len - 2

            if by_word:
                src_corpora = [word for word in src_sent if self.swf.filter(word)]
                if len(src_corpora) > max_len:
                    src_corpora = src_corpora[:max_len]
            else:
                src_corpora = self.sent2corpora(src_sent)
                if len(src_corpora) > max_len:
                    src_corpora = src_corpora[:max_len]

            tgt_corpora = self.sent2corpora(tgt_sent)
            if len(tgt_corpora) > max_len:
                tgt_corpora = tgt_corpora[:max_len]

            src_sents.append([const.WORD[const.BOS]] + src_corpora + [const.WORD[const.EOS]])
            tgt_sents.append([const.WORD[const.BOS]] + tgt_corpora + [const.WORD[const.EOS]])

        src_ignore = self.src_dict.build_idx(src_sents, self._min_word_count)
        tgt_ignore = self.tgt_dict.build_idx(tgt_sents, self._min_word_count)

        if src_ignore != 0:
            logging.info("Ignored src corpus counts - [{}]".format(src_ignore))
        if tgt_ignore != 0:
            logging.info("Ignored tgt corpus counts - [{}]".format(tgt_ignore))

        self.src_sents = src_sents
        self.tgt_sents = tgt_sents

    def parse_valid(self, by_word=True):
        """
        question answer1 answer2...
        """
        src_sents, tgt_sents = [], []
        for sentences in sopen(self._valid_src):
            sentence = sentences.strip().split()
            if len(sentence) != 2: continue
            src_sent, tgt_sent = sentence
            max_len = self._max_len - 2

            if by_word:
                src_corpora = [word for word in src_sent if self.swf.filter(word)]
                if len(src_corpora) > max_len:
                    src_corpora = src_corpora[:max_len]
            else:
                src_corpora = self.sent2corpora(src_sent)
                if len(src_corpora) > max_len:
                    src_corpora = src_corpora[:max_len]

            tgt_corpora = self.sent2corpora(tgt_sent)
            if len(tgt_corpora) > max_len:
                tgt_corpora = tgt_corpora[:max_len]

            src_sents.append([const.WORD[const.BOS]] + src_corpora + [const.WORD[const.EOS]])
            tgt_sents.append([const.WORD[const.BOS]] + tgt_corpora + [const.WORD[const.EOS]])

        self.src_valid_sents = src_sents
        self.tgt_valid_sents = tgt_sents

    def save(self):
        data = {
            'max_token_seq_len': self._max_len,
            'dict': {
                'src': self.src_dict.ind2idx,
                'src_size': len(self.src_dict),
                'tgt': self.tgt_dict.ind2idx,
                'tgt_size': len(self.tgt_dict)
            },
            'train': {
                'src': corpora2idx(self.src_sents, self.src_dict.ind2idx),
                'tgt': corpora2idx(self.tgt_sents, self.tgt_dict.ind2idx)
            }
        }

        if self._valid_src is not None:
            data['valid'] = {
                'src': corpora2idx(self.src_valid_sents, self.src_dict.ind2idx),
                'tgt': corpora2idx(self.tgt_valid_sents, self.tgt_dict.ind2idx)
            }

        torch.save(data, self._save_data)
        logging.info('Finish dumping the corora data to file - [{}]'.format(self._save_data))
        logging.info('src corpora length - [{}] | target corpora length - [{}]'.format(len(self.src_dict), len(self.tgt_dict)))

    def process(self):
        self.parse_train()
        if self._valid_src is not None:
            self.parse_valid()
        self.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='seq2sqe corpora handle')
    parser.add_argument('--train-src', type=str, required=True,
                        help='train file')
    parser.add_argument('--save-data', type=str, required=True,
                        help='path to save processed data')
    parser.add_argument('--valid-src', type=str, default=None,
                        help='valid file')
    parser.add_argument('--max-lenth', type=int, default=32,
                        help='max length of sentence [default: 32]')
    parser.add_argument('--min-word-count', type=int, default='5',
                        help='min corpora count to discard [default: 5]')
    args = parser.parse_args()
    corpus = Corpus(args.train_src, args.save_data,
                    args.valid_src, args.max_lenth, args.min_word_count)
    corpus.process()
