import json
import re
import os

import torch

from const import *

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def words2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]

class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[BOS]: BOS,
            WORD[EOS]: EOS,
            WORD[PAD]: PAD,
            WORD[UNK]: UNK
        }
        self.idx = 4

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

class Captions(object):
    """
    Data: COCO2017
    captions_train2017|captions_val2017 data strcut:
        {"info": info,
        "licenses": [license],
        "images": [image],
        "annotations": [annotation]}
    """
    def __init__(self, cap_path,
            save_data="data/img_caption.pt", max_len=16, min_word_count=2):
        self.dict = Dictionary()
        self.max_len = max_len
        self.min_word_count = min_word_count
        self.save_data = save_data

        self.train_imgs, self.train_labels, cut_counts = self._build(
                "{}/captions_train2017.json".format(cap_path))

        self.val_imgs, self.val_labels, _ = self._build(
                "{}/captions_val2017.json".format(cap_path))

        ignore = self.dict(self.train_labels, self.min_word_count)

        print("Caption`s length out of range numbers - [{}]".format(cut_counts))
        if ignore != 0:
            print("Ignored word counts - [{}]".format(ignore))

        self._save()

    def _build(self, cap_file):
        def _cut(s, cut_counts, max_len):
            words = [w for w in normalizeString(s).strip().split()]
            if len(words) > max_len:
                cut_counts[0] += 1
                words = words[:max_len]
            words = words + [WORD[EOS]]

            return words

        caps = json.loads(next(open(cap_file)))
        images, annotations = caps["images"], caps["annotations"]
        img_dict = {img["id"]: img["file_name"] for img in images}

        cut_counts, imgs, labels = [0], [], []
        for anno in annotations:
            imgs.append(img_dict[anno["image_id"]])
            labels.append(_cut(anno["caption"], cut_counts, self.max_len))

        return imgs, labels, cut_counts[0]

    def _save(self):
        data = {
            'max_word_len': self.max_len+1,
            'vocab_size': len(self.dict),
            'dict': self.dict.word2idx,
            'train': {
                'imgs': self.train_imgs,
                'captions': words2idx(self.train_labels, self.dict.word2idx),
            },
            'valid': {
                'imgs': self.val_imgs,
                'captions': words2idx(self.val_labels, self.dict.word2idx),
            }
        }

        torch.save(data, self.save_data)
        print('words length - [{}]'.format(len(self.dict)))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Image Caption')
    parser.add_argument('--cap_path', type=str, default='data')
    args = parser.parse_args()

    Captions(args.cap_path)