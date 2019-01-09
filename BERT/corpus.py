import torch

from const import *


def word2idx(sents, word2idx):
    results = []
    for t1, t2 in sents:
        t1 = [word2idx[w] if w in word2idx else UNK for w in t1]
        t2 = [word2idx[w] if w in word2idx else UNK for w in t2]
        results.append([t1, t2])

    return results


class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK,
            WORD[SEP]: SEP,
            WORD[CLS]: CLS,
            WORD[MASK]: MASK
        }
        self.idx = len(self.word2idx)

    def add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def __call__(self, sents, min_count=5):
        words = [word for sent in sents for word in sent[0] + sent[1]]
        word_count = {w: 0 for w in set(words)}
        for w in words:
            word_count[w] += 1

        ignored_word_count = 0
        for word, count in word_count.items():
            if count <= min_count:
                ignored_word_count += 1
                continue
            self.add(word)

        return ignored_word_count

    def __len__(self):
        return self.idx


class Corpus(object):
    def __init__(self, save_data="data/corpus.pt", max_len=128):

        self.train = "data/fuel.cnn"
        self.save_data = save_data
        self.word = Dictionary()
        self.max_len = max_len

    def parse_data(self, _file):
        sents = []
        for sentence in open(_file):
            t1, t2 = sentence.strip().split(SPLIT_CODE)

            words1 = t1.strip().split()
            words2 = t2.strip().split()

            sents.append([words1, words2])

        print(f"ignored word count: {self.word(sents)}")
        self.sents = sents

    def save(self):
        self.parse_data(self.train)

        data = {
            'max_len': self.max_len,
            'dict': self.word.word2idx,
            'word': word2idx(self.sents, self.word.word2idx),
        }

        torch.save(data, self.save_data)
        print(f'Finish dumping the data to file - {self.save_data}')
        print(f'words length - {len(self.word)}')


if __name__ == "__main__":
    corpus = Corpus()
    corpus.save()
