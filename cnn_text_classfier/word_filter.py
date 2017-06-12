import os
import os.path as path
_root = path.normpath("%s/.." % path.dirname(path.abspath(__file__)))
import sys
sys.path.append(_root)

def load_words_as_set(filename, enc='utf8'):
    s = set(' ')
    for line in open(filename, 'rb'):
        fields = line.split()
        if len(fields) > 0:
            s.add(fields[0])
    return s

class StopwordFilter(object):
  def __init__(self, filename):
    self.stopwords = load_words_as_set(filename)

  def filter(self, word):
    if isinstance(word, tuple):
      return word[0] not in self.stopwords
    return word not in self.stopwords

class WordLenFilter(object):
  def __init__(self, max_word_len=10):
    self.max_word_len = max_word_len

  def filter(self, word):
    if isinstance(word, tuple):
      return len(word[0]) <= self.max_word_len
    return len(word) <= self.max_word_len
