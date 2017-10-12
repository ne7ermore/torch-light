from common.utils import *
import const

def prepare(doc):
    doc = utterance_preprocess(doc)
    doc = remove_urls(doc)
    doc = remove_emails(doc)
    doc = remove_images(doc)
    return zh_abc_n(doc)

def corpora2idx(sents, ind2idx):
    return [[ind2idx[w] if w in ind2idx else const.UNK for w in s] for s in sents]

