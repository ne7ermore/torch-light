## Introduction
This is the implementation Chinese text multi-classfication

## Requirement
* python 3.5
* pytorch 0.2.0
* Jieba

## Usage
```
python3 main.py -h
```

You will get:

```
usage: main.py [-h] [--lr LR] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
               [--log-interval LOG_INTERVAL] [--save SAVE] [--data DATA]
               [--max-len MAX_LEN] [--dropout DROPOUT] [--embed-dim EMBED_DIM]
               [--kernel-num KERNEL_NUM] [--filter-sizes FILTER_SIZES]
               [--hidden-size HIDDEN_SIZE]
               [--dropout-switches DROPOUT_SWITCHES] [--seed SEED]

CNN text classificer

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               initial learning rate [default: 0.001]
  --epochs EPOCHS       number of epochs for train [default: 256]
  --batch-size BATCH_SIZE
                        batch size for training [default: 50]
  --log-interval LOG_INTERVAL
                        report interval [default: 1000]
  --save SAVE           path to save the final model
  --data DATA           location of the data corpus
  --max-len MAX_LEN     max length of one comment
  --dropout DROPOUT     the probability for dropout (0 = no dropout) [default:
                        0.5]
  --embed-dim EMBED_DIM
                        number of embedding dimension [default: 256]
  --kernel-num KERNEL_NUM
                        number of each kind of kernel
  --filter-sizes FILTER_SIZES
                        filter sizes
  --hidden-size HIDDEN_SIZE
                        hidden size
  --dropout-switches DROPOUT_SWITCHES
                        dropout-switches
  --seed SEED           random seed
```

## Train
```
python3 main.py
```
