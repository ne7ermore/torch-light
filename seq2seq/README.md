## Seq2seq for Question-to-Answer
Module implemention from "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017).

##### Instead of CNN of RNN, reasons for using attention:
* Total computational complexity per layer
* The amount of computation can be parallelized

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>

## Requirement
* python 3.5
* pytorch 0.2.0
* numpy 1.13.1
* Jieba

## Usage

### Prepare training data
```
python3 corpus.py -h
```

You will get:

```
usage: corpus.py [-h] --train-src TRAIN_SRC --save-data SAVE_DATA
                 [--valid-src VALID_SRC] [--max-lenth MAX_LENTH]
                 [--min-word-count MIN_WORD_COUNT]

seq2sqe corpora handle

optional arguments:
  -h, --help            show this help message and exit
  --train-src TRAIN_SRC
                        train file
  --save-data SAVE_DATA
                        path to save processed data
  --valid-src VALID_SRC
                        valid file
  --max-lenth MAX_LENTH
                        max length of sentence [default: 32]
  --min-word-count MIN_WORD_COUNT
                        min corpora count to discard [default: 5]
```

### Train model
```
python3 train.py -h
```

You will get:

```
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--dropout DROPOUT] [--seed SEED]
                [--log-interval LOG_INTERVAL] [--cuda-able] [--not-by-word]
                [--save SAVE] [--save-epoch] [--data DATA]
                [--eval-data EVAL_DATA] [--proj-share-weight]
                [--embs-share-weight] [--d-model D_MODEL]
                [--d-inner-hid D_INNER_HID] [--n-head N_HEAD]
                [--n-layers N_LAYERS] [--n-warmup-steps N_WARMUP_STEPS]

seq2seq

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs for train [default: 32]
  --batch-size BATCH_SIZE
                        batch size for training [default: 64]
  --dropout DROPOUT     the probability for dropout (0 = no dropout) [default:
                        0.1]
  --seed SEED           random seed
  --log-interval LOG_INTERVAL
                        report interval [default: 1000]
  --cuda-able           enables cuda
  --not-by-word         segment sentences not by word
  --save SAVE           path to save the final model
  --save-epoch          save every epoch
  --data DATA           location of the data corpus
  --eval-data EVAL_DATA
                        location of the eval data corpus
  --proj-share-weight   share linear weight
  --embs-share-weight
  --d-model D_MODEL     equal dimension of word embedding dim
  --d-inner-hid D_INNER_HID
  --n-head N_HEAD
  --n-layers N_LAYERS
  --n-warmup-steps N_WARMUP_STEPS
```
example
> python3 train.py --proj-share-weight --data data/seq2seq_train.pt --cuda-able --save-epoch --dropout=0.3

### Predict
```
python3 predict.py
```
You can get [example](https://github.com/ne7ermore/torch_light/blob/master/seq2seq/predict.py#L198)
