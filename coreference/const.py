DATAPATH = "./data"

PAD = 0
UNK = 1

WORD = {
    PAD: '<pad>',
    UNK: '<unk>',
}

INIT_RANGE = 0.1

NORMALIZE_DICT = {"/.": ".", "/?": "?",
                  "-LRB-": "(", "-RRB-": ")",
                  "-LCB-": "{", "-RCB-": "}",
                  "-LSB-": "[", "-RSB-": "]"}
REMOVED_CHAR = ["/", "%", "*"]

FILTERFILES = [
    "cbs_0146.conll",
    "cbs_0166.conll",
    "cnr_0011.conll",
    "cnr_0014.conll",
    "cnr_0024.conll",
    "cnr_0032.conll",
    "cnr_0052.conll",
    "cnr_0092.conll",
    "cnr_0093.conll",
    "cts_0033.conll",
    "ctv_0146.conll",
    "vom_0106.conll",
    "vom_0231.conll",
    "ch_0020.conll",
    "ch_0020.conll",
    "cnr_0060.conll",
]
