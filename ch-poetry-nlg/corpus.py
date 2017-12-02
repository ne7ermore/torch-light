import torch

from const import *

def word2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]

class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[UNK]: UNK
        }
        self.idx = len(self.word2idx)

    def add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def __call__(self, sents, min_count):
        words = [word for sent in sents for word in sent]
        word_count = {w: 0 for w in set(words)}
        for w in words: word_count[w]+=1

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
    def __init__(self, save_data, max_len=20, min_word_count=0):
        self._save_data = save_data
        self._max_len = max_len
        self._min_word_count = min_word_count
        self.sents = None
        self.valid_sents = None
        self.dict = Dictionary()
        self.is_ch = lambda w: (w >= '\u4e00' and w<='\u9fa5') or w == " "

    def parse(self):
        sents, ignore_count = [], 0
        for sentences in open("data/poetry"):
            sentences = sentences.strip()
            words = [w for w in sentences if self.is_ch(w)]
            if len(words) != self._max_len:
                ignore_count += 1
                continue

            sents.append(words)

        print("Data`s length not eq {} - [{}]".format(self._max_len, ignore_count))
        print("Data`s length eq {} - [{}]".format(self._max_len, len(sents)))

        word_ignore = self.dict(sents, self._min_word_count)

        if word_ignore != 0:
            print("Ignored word counts - [{}]".format(word_ignore))

        self.sents = sents

    def save(self):
        data = {
            'max_word_len': self._max_len,
            'dict': {
                'src': self.dict.word2idx,
                'src_size': len(self.dict),
            },
            'train': word2idx(self.sents, self.dict.word2idx)
        }

        torch.save(data, self._save_data)
        print('word length - [{}]'.format(len(self.dict)))

    def process(self):
        self.parse()
        self.save()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Chinese Poetry NLG')
    parser.add_argument('--save-data', type=str, default='data/ch_pro_nlg.pt',
                        help='path to save processed data')
    args = parser.parse_args()
    corpus = Corpus(args.save_data)
    corpus.process()
