## Bidirectional LSTM-CRF NER
Module implemention from "[Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)" (Zhiheng Huang, Wei Xu, Kai Yu Submitted on 9 Aug 2015).

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

#### Accuracy
> 99.8%

##### Example
```
>>> print(predict.get_size("身高170体重55买那个码衣服好？"))

{'height': '170', 'weight': '55'}
```

```
>>> print(predict.get_size("170110斤，穿什么型号"))

{'height': '170', 'weight': '110斤'}
```

```
>>> print(predict.get_size("对啊！我就是体重174斤"))

{'weight': '174斤'}
```
