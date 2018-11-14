import re
import zipfile

from const import SPLIT_CODE


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-zA-Z]+", r"", s)
    return s


def parse_sent(wfile, sents):
    context = []
    for sent in sents:
        temp = []
        for word in sent:
            word = normalizeString(word.decode("utf-8"))
            if len(word) != 0:
                temp += [word]
        context.append(" ".join(temp))

    if len(context) > 1:
        for index in range(len(context) - 1):
            wfile.write(
                SPLIT_CODE.join([context[index], context[index + 1]]) + "\r\n")


def fuel(inf, stop_tag=b"@highlight"):
    zf = zipfile.ZipFile(inf)
    namelist = [fname for fname in zf.namelist() if fname.find(
        ".story") > 0 and fname.find("__MACOSX") == -1]
    print(len(namelist))
    with open("data/fuel.cnn", "w") as wf:
        for name in namelist[:10000]:
            contexts = []
            for line in zf.open(name):
                content = line.strip()
                if content == stop_tag:
                    break

                if len(content) == 0:
                    continue

                sent = line.split()
                contexts.append(sent)

            parse_sent(wf, contexts)


if __name__ == "__main__":
    fuel("data/stories.zip")
