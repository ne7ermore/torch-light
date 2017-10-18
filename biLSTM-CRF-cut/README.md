## BiLSTM CRF Chinese Word Segment

## Requirement
* python 3.5
* pytorch 0.2.0
* numpy 1.13.1

### Train model
```
python3 train.py -h
```

You will get:

```
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--seed SEED]
                [--cuda-able] [--save SAVE] [--save-epoch] [--data DATA]
                [--embed-dim EMBED_DIM] [--dropout DROPOUT]
                [--lstm-hsz LSTM_HSZ] [--lstm-layers LSTM_LAYERS]
                [--w-init W_INIT] [--n-warmup-steps N_WARMUP_STEPS]

CNN Ranking

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs for train [default: 32]
  --batch-size BATCH_SIZE
                        batch size for training [default: 64]
  --seed SEED           random seed
  --cuda-able           enables cuda
  --save SAVE           path to save the final model
  --save-epoch          save every epoch
  --data DATA           location of the data corpus
  --embed-dim EMBED_DIM
                        number of embedding dimension [default: 256]
  --dropout DROPOUT     the probability for dropout (0 = no dropout) [default:
                        0.3]
  --lstm-hsz LSTM_HSZ   BiLSTM hidden size
  --lstm-layers LSTM_LAYERS
                        biLSTM layer numbers
  --w-init W_INIT       weight init scope
  --n-warmup-steps N_WARMUP_STEPS
```

example
> python3 train.py

#### Accuracy
> 96%

##### Segment Example
```
>>> print(sg.sentence_cut("ngram是自然语言处理中一个非常重要的概念，通常在NLP中，
        人们基于一定的语料库，可以利用ngram来预计或者评估一个句子是否合理。
        另外一方面，ngram的另外一个作用是用来评估两个字符串之间的差异程度。
        这是模糊匹配中常用的一种手段。本文将从此开始，
        进而向读者展示ngram在自然语言处理中的各种powerful的应用"))

['ngram', '是', '自然语言', '处理', '中', '一个', '非常', '重要', '的', '概念',
 '，', '通常', '在', 'NLP', '中', '，', '人们', '基于', '一定', '的', '语料库',
 '，', '可以', '利用', 'ngram', '来', '预计', '或者', '评估', '一个', '句子',
 '是否', '合理', '。', '另外', '一方面', '，', 'ngram', '的', '另外', '一个',
 '作用', '是', '用来', '评估', '两个', '字符串', '之间', '的', '差异', '程度',
 '。', '这是', '模糊匹配', '中', '常用', '的', '一种', '手段', '。', '本文', '将',
  '从此', '开始', '，', '进而', '向', '读者', '展示', 'ngram', '在', '自然语言',
  '处理', '中', '的', '各种', 'powerful', '的', '应用']
```
