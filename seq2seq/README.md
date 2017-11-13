## Seq2seq for French to English translation
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

## Train data
The data for this project is a set of many thousands of French to English translation pairs. <br>
Downloads available at http://www.manythings.org/anki/

## Usage

### Translate

```
python transform.py --French="Tourne à droite au prochain carrefour."

Translated - turn right at the next stayed .
```
*****
```
python transform.py --French="Il est de ta responsabilité de terminer ce travail."

Translated - it s your responsibility to finish the job .
```
*****
```
python transform.py --French="J'ai besoin d'une paire de ciseaux pour couper ce papier."

Translated - i need a pair of scissors to cut this paper .
```

### Train model
```
python3 train.py -h
```

You will get:

```
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--seed SEED]
                [--cuda-able] [--save SAVE] [--data DATA] [--not-share-linear]
                [--dropout DROPOUT] [--d-model D_MODEL] [--d-ff D_FF]
                [--n-head N_HEAD] [--n-stack-layers N_STACK_LAYERS]
                [--n-warmup-steps N_WARMUP_STEPS]

seq2seq

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs for train
  --batch-size BATCH_SIZE
                        batch size for training
  --seed SEED           random seed
  --cuda-able           enables cuda
  --save SAVE           path to save the final model
  --data DATA           location of the data corpus
  --not-share-linear    Share the weight matrix between tgt word
                        embedding/linear
  --dropout DROPOUT     the probability for dropout (0 = no dropout)
  --d-model D_MODEL     equal dimension of word embedding dim
  --d-ff D_FF           Position-wise Feed-Forward Networks inner layer dim
  --n-head N_HEAD
  --n-stack-layers N_STACK_LAYERS
  --n-warmup-steps N_WARMUP_STEPS
```

