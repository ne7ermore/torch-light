import torch

from const import *


def reps2idx(responses, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in rep] for rep in responses]


def uttes2idx(utterances, word2idx):
    return [[[word2idx[w] if w in word2idx else UNK for w in u] for u in utte] for utte in utterances]


class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK,
        }
        self.idx = len(self.word2idx)

    def add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def __call__(self, utterances, responses, min_count):
        words = [word for resp in responses for word in resp]
        words += [word for utte in utterances for u in utte for word in u]

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

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))


class Corpus(object):
    def __init__(self, max_cont_len=10, max_utte_len=50, min_word_count=2):
        self.dict = Dictionary()
        self.max_cont_len = max_cont_len
        self.max_utte_len = max_utte_len
        self.min_word_count = min_word_count

        self.parse_data("data/dev.txt", False)
        self.parse_data("data/train.txt", True)
        self.save()

    def parse_data(self, inf, is_train):
        utterances, responses, labels = [], [], []

        for line in open(inf):
            contexts = line.strip().split("\t")
            uttes, resp, l = contexts[1:-1], contexts[-1], contexts[0]

            resp = resp.split()
            uttes = [utte.split() for utte in uttes]

            if len(resp) > self.max_utte_len:
                resp = resp[:self.max_utte_len]

            if len(uttes) > self.max_cont_len:
                # close to response
                uttes = uttes[-self.max_cont_len:]

            for index, utte in enumerate(uttes):
                if len(utte) > self.max_utte_len:
                    uttes[index] = utte[:self.max_utte_len]

            utterances.append(uttes)
            responses.append(resp)
            labels.append(int(l))

        if is_train:
            ignore_w_nums = self.dict(
                utterances, responses, self.min_word_count)
            self.train_utterances = utterances
            self.train_responses = responses
            self.train_labels = labels

            print("Ignored counts - [{}]".format(ignore_w_nums))

        else:
            self.test_utterances = utterances
            self.test_responses = responses
            self.test_labels = labels

    def save(self):
        data = {
            'max_cont_len': self.max_cont_len,
            'max_utte_len': self.max_utte_len,
            'dict': {
                'dict': self.dict.word2idx,
                'dict_size': len(self.dict),
            },
            'train': {
                'responses': reps2idx(self.train_responses, self.dict.word2idx),
                'utterances': uttes2idx(self.train_utterances, self.dict.word2idx),
                'labels': self.train_labels
            },
            'test': {
                'responses': reps2idx(self.test_responses, self.dict.word2idx),
                'utterances': uttes2idx(self.test_utterances, self.dict.word2idx),
                'labels': self.test_labels
            }
        }

        torch.save(data, "data/corpus")
        print('dict length - [{}]'.format(len(self.dict)))


if __name__ == "__main__":
    Corpus()
