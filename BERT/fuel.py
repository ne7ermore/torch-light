import re
import zipfile

from const import SPLIT_CODE


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-zA-Z.!?]+", r"", s)
    s = re.sub(r"([.!?])", r" \1", s)
    return s


def parse_sent(wfile, sent, min_len=5):
    temp, sents = [], []
    for word in sent:
        word = normalizeString(word.decode("utf-8"))
        temp += [word]
        if re.search(r"[.!?]", word) is not None:
            sents.append(" ".join(temp))
            temp = []

    if len(sents) > 1:
        for index in range(len(sents) - 1):
            wfile.write(SPLIT_CODE.join(
                [sents[index], sents[index + 1]]) + "\r\n")


def fuel(inf, stop_tag=b"\r\n", min_len=20):
    zf = zipfile.ZipFile(inf)
    namelist = [fname for fname in zf.namelist() if fname.find(".txt") > 0]

    contexts = []
    with open("data/fuel", "w") as wf:
        for name in namelist[:100]:
            for line in zf.open(name):
                if line == stop_tag and len(contexts) > 0:
                    if len(contexts) > min_len:
                        parse_sent(wf, contexts)
                    contexts = []

                words = line.strip().split()
                contexts += words


if __name__ == "__main__":
    fuel("data/Gutenberg.zip")
