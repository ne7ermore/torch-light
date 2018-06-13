import gzip
import json
import random
import sys
import re

import pandas as pd


def process(inf):
    def normalizeString(s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

        return s

    datas = []
    g = gzip.open(inf, "r")
    for l in g:
        js = json.loads(json.dumps(eval(l)))
        datas.append((js["reviewText"], js["summary"], js["overall"]))

    random.shuffle(datas)

    columns = ["original", "summary", "score"]
    train = pd.DataFrame(datas[len(datas) // 20:], columns=columns)
    test = pd.DataFrame(datas[:len(datas) // 20], columns=columns)

    for df in [train, test]:
        df["original"] = df["original"].apply(lambda x: normalizeString(x))
        df["summary"] = df["summary"].apply(lambda x: normalizeString(x))
        df["score"] = df["score"].apply(lambda x: int(x))

    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python3 fuel.py toys | sports | movies')
        exit()

    dataset = sys.argv[1]

    if dataset == "toys":
        inf = "data/reviews_Toys_and_Games_5.json.gz"
    elif dataset == "sports":
        inf = "data/reviews_Sports_and_Outdoors_5.json.gz"
    elif dataset == "movies":
        inf = "data/reviews_Movies_and_TV_5.json.gz"
    else:
        print('Usage: python3 fuel.py toys | sports | movies')
        exit()

    process(inf)
