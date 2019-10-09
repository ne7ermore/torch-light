import torch
import json
import os
import collections

import jieba.posseg as pseg

import const
import common


class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            const.WORD[const.PAD]: const.PAD,
            const.WORD[const.UNK]: const.UNK,
        }
        self.idx = len(self.word2idx)

    def add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def parse(self, texts, min_count=0):
        words = [word for text in texts for word in text]

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
    def __init__(self, max_len=512, load_w2v=None, save_data="data/corpus.pt"):

        self.save_data = save_data
        self.load_w2v = load_w2v
        self.word = Dictionary()
        self.char = Dictionary()
        self.max_len = max_len
        self.seg = pseg
        self.parse_pred()
        self.parse_data(True)
        self.parse_data(False)
        self.save()

    def parse_pred(self):
        predicate2id = {}
        lines = open("data/all_50_schemas", encoding="utf8")
        for line in lines:
            line = json.loads(line)
            if line["predicate"] not in predicate2id:
                predicate2id[line["predicate"]] = len(predicate2id)

        self.predicate2id = predicate2id
        lines.close()

    def segment(self, text):
        words = []
        for w in [[e]*len(e) for e, _ in self.seg.cut(text)]:
            words += w

        return words

    def parse_data(self, is_train):
        chars, words = [], []
        sub_sidx, sub_eidx = [], []
        sub_slidx, sub_elidx = [], []
        obj_idxs = []

        if is_train:
            inf = open("data/train_data.json", encoding="utf8")
        else:
            inf = open("data/dev_data.json", encoding="utf8")

        for line in inf:
            line = json.loads(line)
            text = line["text"]
            if len(text) > self.max_len:
                text = text[:self.max_len]

            items = collections.defaultdict(list)

            # subject position
            s1, s2 = [0]*len(text), [0]*len(text)

            for spo in line['spo_list']:
                subject, predicate, obj = spo["subject"].strip('《》').strip().lower(
                ), spo["predicate"], spo["object"].strip('《》').strip().lower()

                subjectid = text.find(subject)
                objectid = text.find(obj)
                if subjectid != -1 and objectid != -1:
                    items[(subjectid, subjectid+len(subject)-1)].append((objectid,
                                                                         objectid+len(obj)-1, self.predicate2id[predicate]))

                    s1[subjectid] = 1
                    s2[subjectid+len(subject)-1] = 1

            if len(items):
                for (sub_s, sub_e), obj_idx in items.items():
                    # text - chars & words
                    chars.append(text)
                    words.append(self.segment(text))
                    assert len(chars[-1]) == len(words[-1])

                    sub_sidx.append(s1)
                    sub_eidx.append(s2)

                    # subject index
                    sub_slidx.append([sub_s])
                    sub_elidx.append([sub_e])
                    obj_idxs.append(obj_idx)

        assert len(chars) == len(words)
        assert len(chars) == len(sub_sidx)
        assert len(chars) == len(sub_eidx)
        assert len(chars) == len(obj_idxs)
        assert len(chars) == len(sub_slidx)
        assert len(chars) == len(sub_elidx)

        if is_train:
            print(f"ignore words count - {self.word.parse(words, 1)}")
            self.char.parse(chars)
            self.train_char2idx = common.question2idx(
                chars, self.char.word2idx)
            self.train_word2idx = common.question2idx(
                words, self.word.word2idx)
            self.train_sub_sidx = sub_sidx
            self.train_sub_eidx = sub_eidx
            self.train_obj_idxs = obj_idxs
            self.train_sub_slidx = sub_slidx
            self.train_sub_elidx = sub_elidx
        else:
            self.dev_char2idx = common.question2idx(chars, self.char.word2idx)
            self.dev_word2idx = common.question2idx(words, self.word.word2idx)
            self.dev_sub_sidx = sub_sidx
            self.dev_sub_eidx = sub_eidx
            self.dev_obj_idxs = obj_idxs
            self.dev_sub_slidx = sub_slidx
            self.dev_sub_elidx = sub_elidx

    def save(self):
        data = {
            'max_len': self.max_len,
            'word2idx': self.word.word2idx,
            'char2idx': self.char.word2idx,
            'predicate2id': self.predicate2id,
            "train": {
                "char": self.train_char2idx,
                "word": self.train_word2idx,
                "sub_sidx": self.train_sub_sidx,
                "sub_eidx": self.train_sub_eidx,
                "obj_idxs": self.train_obj_idxs,
                "sub_slidx": self.train_sub_slidx,
                "sub_elidx": self.train_sub_elidx,
            },
            "dev": {
                "char": self.dev_char2idx,
                "word": self.dev_word2idx,
                "sub_sidx": self.dev_sub_sidx,
                "sub_eidx": self.dev_sub_eidx,
                "obj_idxs": self.dev_obj_idxs,
                "sub_slidx": self.dev_sub_slidx,
                "sub_elidx": self.dev_sub_elidx,
            }
        }

        if self.load_w2v is not None:
            charW, wordW = common.load_pre_w2c(
                self.load_w2v, self.char.word2idx, self.word.word2idx)
            data["charW"] = charW
            data["wordW"] = wordW

        torch.save(data, self.save_data)
        print(f'train data length - {len(self.train_char2idx)}')
        print(f'dev data length - {len(self.dev_char2idx)}')
        print(f'char length - {len(self.char.word2idx)}')
        print(f'word length - {len(self.word.word2idx)}')
        print(f'Finish dumping the data to file - {self.save_data}')


if __name__ == "__main__":
    Corpus()
