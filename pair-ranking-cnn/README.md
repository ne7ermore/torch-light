## Pair Ranking CNN
Module implemention from [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf)

## Requirement
* python 3.5
* pytorch 0.2.0
* numpy 1.13.1

## Usage

### Prepare training data
```
python3 corpus.py -h
```

You will get:

```
usage: corpus.py [-h] --train-src TRAIN_SRC --save-data SAVE_DATA
                 [--valid-src VALID_SRC] [--max-lenth-src MAX_LENTH_SRC]
                 [--max-lenth-tgt MAX_LENTH_TGT]
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
  --max-lenth-src MAX_LENTH_SRC
                        max length left of sentence [default: 32]
  --max-lenth-tgt MAX_LENTH_TGT
                        max length right of sentence [default: 32]
  --min-word-count MIN_WORD_COUNT
                        min corpora count to discard [default: 1]
```

### Train model
```
python3 train.py -h
```

You will get:

```
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--seed SEED]
                [--log-interval LOG_INTERVAL] [--cuda-able] [--lr LR]
                [--save SAVE] [--save-epoch] [--data DATA]
                [--embed-dim EMBED_DIM] [--filter-sizes FILTER_SIZES]
                [--num-filters NUM_FILTERS] [--dropout DROPOUT]
                [--hidden-size HIDDEN_SIZE] [--l_2 L_2]

CNN Ranking

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs for train [default: 32]
  --batch-size BATCH_SIZE
                        batch size for training [default: 64]
  --seed SEED           random seed
  --log-interval LOG_INTERVAL
                        report interval [default: 1000]
  --cuda-able           enables cuda
  --lr LR               initial learning rate [default: 0.001]
  --save SAVE           path to save the final model
  --save-epoch          save every epoch
  --data DATA           location of the data corpus
  --embed-dim EMBED_DIM
                        number of embedding dimension [default: 64]
  --filter-sizes FILTER_SIZES
                        filter sizes
  --num-filters NUM_FILTERS
                        Number of filters per filter size [default: 64]
  --dropout DROPOUT     the probability for dropout (0 = no dropout) [default:
                        0.5]
  --hidden-size HIDDEN_SIZE
                        hidden size
  --l_2 L_2             L2 regularizaion lambda [default: 0.0]
```

example
> python3 train.py

### Predict
```
python3 predict.py
```
