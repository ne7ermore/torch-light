import io
from collections import defaultdict
from copy import deepcopy
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

import const

def load_pre_w2c(_file, word2idx, char2idx):
    w2c_dict = {}
    print("loading word2vec")
    for line in open(_file):
        temp = line.strip().split(" ")

        if len(temp) < 10:
            continue
        w2c_dict[temp[0]] = list(map(float, temp[1:]))

        if "len_" not in locals():
            len_ = len(temp[1:])

    print(f"load {len(w2c_dict)} lines word2vec")

    wordW = np.random.rand(len(word2idx), len_)
    for word, idx in sorted(word2idx.items(), key=lambda x: x[1]):
        if word in w2c_dict:
            wordW[idx] = np.asarray(w2c_dict[word])

    charW = np.random.rand(len(char2idx), len_)
    for word, idx in sorted(char2idx.items(), key=lambda x: x[1]):
        if word in w2c_dict:
            charW[idx] = np.asarray(w2c_dict[word])

    return wordW, charW

def is_chinese_char(c):
    if ((c >= 0x4E00 and c <= 0x9FFF) or
            (c >= 0x3400 and c <= 0x4DBF) or
            (c >= 0x20000 and c <= 0x2A6DF) or
            (c >= 0x2A700 and c <= 0x2B73F) or
            (c >= 0x2B740 and c <= 0x2B81F) or
            (c >= 0x2B820 and c <= 0x2CEAF) or
            (c >= 0xF900 and c <= 0xFAFF) or
            (c >= 0x2F800 and c <= 0x2FA1F)):
        return True
    return False


def clean_token(token):
    if token in ['.', '?', '!', "。", '？', '！']:
        return token

    token = ''.join(c for c in token if is_chinese_char(ord(c)))
    if token:
        return token

    return const.WORD[const.UNK]

def load_file(filename):
    """ Load a *._conll file
    Input:
        filename: path to the file
    Output:
        documents: list of Document class for each document in the file containing:
            tokens:                   split list of text
            utts_corefs:
                coref['label']:     id of the coreference cluster
                coref['start']:     start index (index of first token in the utterance)
                coref['end':        end index (index of last token in the utterance)
                coref['span']:      corresponding span
            utts_speakers:          list of speakers
            genre:                  genre of input
    """
    documents = []
    with io.open(filename, 'rt', encoding='utf-8', errors='strict') as f:
        tokens, text, utts_corefs, corefs, index = [], [], [], [], 0
        for line in f:
            cols = line.split()

            # End of utterance within a document: update lists, reset variables for next utterance.
            if len(cols) == 0:
                if text:
                    tokens.extend(text), utts_corefs.extend(corefs)
                    text, corefs = [], []
                    continue

            # End of document: organize the data, append to output, reset variables for next document.
            elif len(cols) == 2:
                doc = Document(tokens, utts_corefs, filename)
                documents.append(doc)
                tokens, text, utts_corefs, index = [], [], [], 0

            # Inside an utterance: grab text, speaker, coreference information.
            elif len(cols) > 7:
                text.append(clean_token(cols[3]))

                # If the last column isn't a '-', there is a coreference link
                if cols[-1] != u'-':
                    coref_expr = cols[-1].split(u'|')
                    for token in coref_expr:

                        # Check if coref column token entry contains (, a number, or ).
                        match = re.match(r"^(\(?)(\d+)(\)?)$", token)
                        label = match.group(2)

                        # If it does, extract the coref label, its start index,
                        if match.group(1) == u'(':
                            corefs.append({'label': label,
                                           'start': index,
                                           'end': None})

                        if match.group(3) == u')':
                            for i in range(len(corefs)-1, -1, -1):
                                if corefs[i]['label'] == label and corefs[i]['end'] is None:
                                    break

                            # Extract the end index, include start and end indexes in 'span'
                            corefs[i].update({'end': index,
                                              'span': (corefs[i]['start'], index)})

                index += 1
            else:

                # Beginning of Document, beginning of file, end of file: nothing to scrape off
                continue

    return documents

class Document(object):

    dots = ['.', '?', '!', "。", '？', '！']
    bins = [1,2,3,4,8,16,32,64]

    def __init__(self, tokens, corefs, filename, max_len=500, span_len=4):
        self.tokens = tokens
        self.corefs = corefs
        self.filename = filename
        self.max_len = max_len
        self.span_len = span_len
        if len(self) > max_len:
            self.tokens = self.tokens[:max_len]

    def __len__(self):
        return len(self.tokens)

    def mentions(self, word2idx):
        core_dict = defaultdict(list)
        for coref in self.corefs:
            if coref["end"] < self.max_len:
                core_dict[coref["label"]].append(coref)        

        self.mentions = []
        for label, mentions in core_dict.items():
            mentions.sort(key=lambda x: x["end"], reverse=True)
            idx = 0
            while idx < len(mentions)-1:
                start_idx = max(mentions[idx]["start"], mentions[idx]["end"]+1-self.span_len)
                if sum([1 if w in word2idx else 0 for w in self.tokens[start_idx: mentions[idx]["end"]+1]]) == 0:
                    idx += 1
                else:
                    mention = mentions[idx]['span']
                    corefs = [m["span"] for m in mentions[idx+1:]]
                    uncorefs = []
                    for other_label, other_mentions in core_dict.items():
                        if other_label != label:
                            uncorefs += [m["span"] for m in other_mentions if m["end"] <= mention[0]]
                    self.mentions.append(Mention(mention, corefs, uncorefs))
                    break
            
    def tokens2tensor(self, use_cuda, word2idx):
        self.token_tensors = torch.LongTensor([word2idx[w] if w in word2idx else const.UNK for w in self.tokens])
        if use_cuda:
            self.token_tensors = self.token_tensors.cuda()

    def pos2tensor(self, use_cuda):
        pos = np.array([pos_i+1 for pos_i in range(len(self))])
        pos = torch.from_numpy(pos)
        if use_cuda:
            pos = pos.cuda()
        return pos.long()

    def span2tonsor(self, word2idx):
        corefs_idxs, mention_idxs, mention_spans, labels, distances, corefs = [], [], [], [], [], []

        for mention in self.mentions:
            mention_start_idx, mention_end_idx = mention.mention_span
            mention_start_idx = max(mention_end_idx + 1 - self.span_len, mention_start_idx)
            mention_idx = (mention_start_idx, mention_end_idx)

            mention_span = [word2idx[w] if w in word2idx else const.UNK for w in self.tokens[
                mention_start_idx: mention_end_idx+1]] + [const.PAD] * (self.span_len-(mention_end_idx-mention_start_idx+1))
            
            for (coref_start_idx, coref_end_idx) in mention.corefs:
                coref_start_idx = max(coref_start_idx, coref_end_idx + 1 - self.span_len)
                if coref_start_idx == mention_start_idx and coref_end_idx == mention_end_idx:
                    continue

                # unk过滤:
                if sum([1 if w in word2idx else 0 for w in self.tokens[coref_start_idx: coref_end_idx+1]]) == 0:
                    continue                

                coref_idx = (coref_start_idx, coref_end_idx)
                coref_span = [word2idx[w] if w in word2idx else const.UNK for w in self.tokens[
                    coref_start_idx: coref_end_idx+1]] + [const.PAD] * (self.span_len-(coref_end_idx-coref_start_idx+1))                
         
                corefs_idxs.append(coref_idx)
                mention_idxs.append(mention_idx)
                mention_spans.append(mention_span)
                labels.append(1)
                corefs.append(coref_span)

                length = mention_start_idx-coref_end_idx+1
                distances.append(sum([True for i in self.bins if length >= i]))

            for (coref_start_idx, coref_end_idx) in mention.uncorefs:
                coref_start_idx = max(coref_start_idx, coref_end_idx + 1 - self.span_len)
                if coref_start_idx == mention_start_idx and coref_end_idx == mention_end_idx:
                    continue

                # unk过滤:
                if sum([1 if w in word2idx else 0 for w in self.tokens[coref_start_idx: coref_end_idx+1]]) == 0:
                    continue                    

                coref_idx = (coref_start_idx, coref_end_idx)
                coref_span = [word2idx[w] if w in word2idx else const.UNK for w in self.tokens[
                    coref_start_idx: coref_end_idx+1]] + [const.PAD] * (self.span_len-(coref_end_idx-coref_start_idx+1))                
         
                corefs_idxs.append(coref_idx)
                mention_idxs.append(mention_idx)
                mention_spans.append(mention_span)
                labels.append(0)
                corefs.append(coref_span)

                length = mention_start_idx-coref_end_idx+1
                distances.append(sum([True for i in self.bins if length >= i]))       

        self.mention_spans = np.asarray(mention_spans)
        self.labels = np.asarray(labels)
        self.corefs = np.asarray(corefs)
        self.distances = np.asarray(distances)
        self.corefs_idxs = np.asarray(corefs_idxs)
        self.mention_idxs = np.asarray(mention_idxs)

    def sample(self, use_cuda, numbers):
        choice = np.random.choice(np.arange(self.mention_spans.shape[0]), min(numbers, self.mention_spans.shape[0]))

        mention_spans = self.mention_spans[choice] 
        labels = self.labels[choice] 
        corefs = self.corefs[choice] 
        distances = self.distances[choice] 
        corefs_idxs = self.corefs_idxs[choice] 
        mention_idxs = self.mention_idxs[choice] 

        mention_spans, labels, distances, corefs = map(torch.from_numpy, (mention_spans, labels, distances, corefs))

        mention_spans, distances, corefs = map(lambda x: x.long(), (mention_spans, distances, corefs))     
        labels = labels.float()   

        if use_cuda:
            mention_spans, labels, distances, corefs = map(lambda x: x.cuda(), (mention_spans, labels, distances, corefs))

        return corefs_idxs, mention_idxs, mention_spans, labels, distances, corefs

class Mention(object):
    def __init__(self, mention_span, corefs, uncorefs):
        self.mention_span = mention_span
        self.corefs = corefs
        self.uncorefs = uncorefs


if __name__ == "__main__":
    import torch, os
    documents = load_file("./data/train/cbs_0001.conll")
    corpus = torch.load(os.path.join(const.DATAPATH, "corpus.pt"))

    doc = documents[0]
    doc.tokens2tensor(False, corpus["word2idx"])
    print(len(doc))    

    idx2word = {v:k for k, v in corpus["word2idx"].items()}
    idxs = doc.token_tensors.tolist()
    print(" ".join([idx2word[idx] for idx in idxs]))

    doc.mentions(corpus["word2idx"])

    print("="*50)
    for mention in doc.mentions:
        print(" ".join(idx2word[idx] for idx in idxs[mention.mention_span[0]:mention.mention_span[1]+1]))
        for core in mention.corefs:
            print(" ".join(idx2word[idx] for idx in idxs[core[0]:core[1]+1]))        
        print("="*50)

    doc.span2tonsor(corpus["word2idx"])

    corefs_idxs, mention_idxs, mention_spans, labels, distances, corefs = doc.sample(False, 1)
    print(corefs_idxs, mention_idxs, distances)