import jieba
import jieba.posseg as pseg
import copy

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

class Jieba(object):

    def __init__(self, dict_dir, useSynonym=False, HMM=True):
        self.jieba = jieba
        self.pseg = pseg
        self.HMM = HMM
        self.jieba.load_userdict(dict_dir + "/default.dict")

        # synonym
        if useSynonym:
            self.synonym = {}
            self.synonym_max_len = 0
            for line in open(dict_dir + "/synonym", mode='rb'):
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
        words = self.pseg.cut(uutf8(text, from_enc='utf8'), HMM=self.HMM)
        return [(rstr(w), f) for w, f in words]

    def segment_search(self, text):
        if not text: return []
        words = self.jieba.cut_for_search(uutf8(text, from_enc='utf8'))
        return [w for w  in words]
