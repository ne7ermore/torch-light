import const

def corpora2idx(sents, ind2idx):
    return [[ind2idx[w] if w in ind2idx else const.UNK for w in s] for s in sents]

