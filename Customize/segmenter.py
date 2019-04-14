import os
import copy
import sys
import gzip
import codecs

import jieba
import jieba.posseg as pseg


def uutf8(s, from_enc='utf8'):
    if isinstance(s, bytes):
        return s
    return s.encode(from_enc)


def rstr(s, from_enc='utf8', to_enc='utf8'):
    if isinstance(s, bytes):
        return s.decode(to_enc)
    if from_enc == to_enc:
        return s
    return uutf8(s, from_enc).decode(to_enc)


def sopen(filename, mode='rb', enc='utf8', errors='strict'):
    readMode = 'r' in mode
    if readMode and 'w' in mode:
        print("Must be either read mode or write, but not both")
        raise

    elif filename.endswith('.gz'):
        stream = gzip.GzipFile(filename, mode)
    elif filename == '-':
        if readMode:
            stream = sys.stdin
        else:
            stream = sys.stdout
    else:
        stream = open(filename, mode)

    if enc not in (None, 'byte'):
        if readMode:
            return codecs.getreader(enc)(stream, errors)
        else:
            return codecs.getwriter(enc)(stream, errors)
    return stream


class Jieba(object):
    def __init__(self, useSynonym=False, HMM=False):

        self.jieba = jieba
        self.pseg = pseg
        self.HMM = HMM
        _path = os.path.normpath(
            "%s/" % os.path.dirname(os.path.abspath(__file__)))
        self.jieba.load_userdict(os.path.join(
            _path, "segmenter_dicts/default.dict"))

        # synonym
        if useSynonym:
            self.synonym = {}
            self.synonym_max_len = 0
            for line in sopen(os.path.join(_path, "segmenter_dicts/synonym")):
                fields = line.strip().split()
                if len(fields) != 2:
                    continue
                self.synonym[fields[0]] = fields[1]
                self.synonym_max_len = max(
                    self.synonym_max_len, len(fields[0]))

        # number
        self.number = {
            "零": '0',
            "一": '1',
            "二": '2',
            "三": '3',
            "四": '4',
            "五": '5',
            "六": '6',
            "七": '7',
            "八": '8',
            "九": '9'}

        # punct
        self.punct = {
            "　": ' ',
            "～": '~'
        }

    def join(self, words, syn=False, num=False, punct=True, period=True):
        words = [w[0] for w in words]
        if syn:
            n = len(words)
            for i in range(n):
                if not words:
                    continue

                m = 0
                j = i
                while j < n:
                    m += len(words[j])
                    if m > self.synonym_max_len:
                        break
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
        if not text:
            return []
        words = self.pseg.cut(uutf8(text, from_enc='utf8'), HMM=self.HMM)
        return [(rstr(w), f) for w, f in words]

    def segment_search(self, text):
        if not text:
            return []
        words = self.jieba.cut_for_search(uutf8(text, from_enc='utf8'))
        return [w for w in words]


if __name__ == '__main__':
    jb = Jieba(useSynonym=True, HMM=False)
    print([e[0] for e in jb.segment("你宅男宅女")])
    print([e[1] for e in jb.segment("刘乐妍跟王宝强很配人同样觉得么")])
    print([e[0] for e in jb.segment("刘乐妍跟王宝强很配人同样觉得么")])
    print([e[0] for e in jb.segment("做人不能太CNN")])
    print([e[0] for e in jb.segment("我是北京派来的")])
    print([e[0] for e in jb.segment("关我吊事")])
    print([e[0] for e in jb.segment("hEllo，鮮花快递催单？")])
    print([e[0] for e in jb.segment("我们中出了一个叛徒")])
    print(jb.segment_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造"))
    print(' '.join(['/'.join(e) for e in jb.segment("hEllo，鮮花快递催单？")]))
    print(' '.join(['/'.join(e) for e in jb.segment("包邮费")]))
    print(' '.join(['/'.join(e) for e in jb.segment("免邮费")]))
    print(' '.join(['/'.join(e) for e in jb.segment("多久几天")]))
    print(jb.join(jb.segment('你的　快递单号是多少，Jason，一百五十七～。'),
                  syn=False, num=False, punct=True))
    print(jb.join(jb.segment("在哈我勒个去在么有少优惠有啥有什么晕解决修复材质质地"),
                  syn=True, num=True, punct=True, period=False))
    print(' '.join([''.join(e[0])
                    for e in jb.segment("redis问题，已解决补充下哈，九彩花田是客户端出现授权说明材质质地")]))
    print(' '.join([''.join(e[0]) for e in jb.segment("请帮我解梦")]))
    print([''.join(e[0])
           for e in jb.segment("redis问题，已解决补充下哈，九彩花田是客户端出现授权说明材质质地")])
