import const

def corpora2idx(sents, ind2idx):
    return [[ind2idx[w] if w in ind2idx else const.UNK for w in s] for s in sents]

class Trie(object):
    def __init__(self):
        self.tree = dict()

    def add(self, s):
        tree = self.tree
        for w in s:
            tree = tree.setdefault(w, dict())
