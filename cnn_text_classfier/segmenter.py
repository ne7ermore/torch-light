""" jieba segmenter
"""
import os
import os.path as path
_root = path.normpath("%s/.." % path.dirname(path.abspath(__file__)))
import sys
sys.path.append(_root)
import jieba
import jieba.posseg as pseg
from common.utils import uutf8, rstr, sopen
import copy

class Jieba(object):

    def __init__(self, dict_dir, useSynonym=False):
        self.jieba = jieba
        self.pseg = pseg
        self.jieba.load_userdict(dict_dir + "/default.dict")

        # synonym
        if useSynonym:
            self.synonym = {}
            self.synonym_max_len = 0
            for line in sopen(dict_dir + "/synonym"):
                fields = line.strip().split()
                if len(fields) != 2: continue
                self.synonym[fields[0]] = fields[1]
                self.synonym_max_len = max(self.synonym_max_len, len(fields[0]))

        # number
        self.number = {
            "零" : '0',
            "一" : '1',
            "二" : '2',
            "三" : '3',
            "四" : '4',
            "五" : '5',
            "六" : '6',
            "七" : '7',
            "八" : '8',
            "九" : '9'}

        # punct
        self.punct = {
            "　" : ' ',
            "～" : '~'
        }

    def join(self, words, syn=False, num=False, punct=True, period=True):
        words = [w[0] for w in words]
        if syn:
            n = len(words)
            # range in Python 3.x is xrange
            for i in range(n):
                if not words: continue

                m = 0
                j = i
                while j < n:
                    m += len(words[j])
                    if m > self.synonym_max_len: break
                    j += 1

                while j > i:
                    compound = u''.join(words[i:j])
                    if compound in self.synonym:
                        compound = self.synonym[compound]
                        words[i] = compound
                        for k in range(i + 1, j):
                            words[k] = u''
                        break
                    j -= 1
        res = u''.join(words)

        if num:
            chs = []
            for ch in res:
                chs.append(self.number.get(ch, ch))
            res = u''.join(chs)

        puncts = copy.deepcopy(self.punct)
        if period:
            puncts[u'。'] = u'.'

        if punct:
            chs = []
            for ch in res:
                chs.append(puncts.get(ch, ch))
            res = u''.join(chs)
        return res

    def segment(self, text):
        if not text: return []
        words = self.pseg.cut(uutf8(text, from_enc='utf8'))
        return [(rstr(w), f) for w, f in words]
