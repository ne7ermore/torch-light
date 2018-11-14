PAD = 0
UNK = 1
SEP = 2
CLS = 3
MASK = 4


WORD = {
    UNK: '<unk>',
    PAD: '<pad>',
    SEP: '<sep>',
    CLS: '<cls>',
    MASK: '<mask>',
}


RANDOM_MARK = 0.8
RANDOM_WORD = 0.5
RANDOM_WORD_SAMPLE = 0.15
RANDOM_SENT = 0.5

NEXT = 1
NOT_NEXT = 0

SEGMENTA = 1
SEGMENTB = 2

INIT_RANGE = 0.02

NOT_USE_WEIGHT_DECAY = ['bias', 'gamma', 'beta']
SPLIT_CODE = "@@@###@@@"
