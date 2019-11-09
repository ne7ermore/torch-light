import os

import const
import skeleton2conll


def skele2conll(data_type):
    for root, _, files in os.walk(os.path.join(const.DATAPATH, "/conll-2012/", data_type)):
        for inf in files:
            if inf.endswith("gold_skel"):
                conll_file = os.path.join(root, inf)
                ldc_file = conll_file.replace(
                    "conll-2012/"+data_type, "ldc").replace("v4_gold_skel", "onf")
                output_file = f'{const.DATAPATH}/data/{data_type}/{inf.replace("v4_gold_skel", "conll")}'
                skeleton2conll.start(ldc_file, conll_file, output_file, "utf8")


if __name__ == "__main__":
    skele2conll("train")
    skele2conll("development")
