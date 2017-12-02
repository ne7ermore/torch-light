## Introduction
Generation for Chinese Poetry.
Module implemention from "[Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349#)". <br>

## Requirement
* python 3.5
* pytorch 0.2.0
* numpy 1.13.1
* tqdm

## Example
```
金史良吾勤，不免慰踟蹰，悠悠沧洲渚，莺香拂玉巾。

苟色涨烟阔，森戟启神功，神凶元元洁，元化威刀威。

云根烟叶影，苔冷白露倾，素尾题琴歌，梅房带林东。

寓者望此海，征溟叠苍漠，漠渚连湘寺，斜吟枕寺枝。

山颇无风如，光骸其混肌，义肃氛旗凝，云旗苍崖涧。
```

## Train
```
python3 train.py
```

Help:

```
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--seed SEED]
                [--unuse-cuda] [--n-warmup-steps N_WARMUP_STEPS] [--save SAVE]
                [--data DATA] [--embed-dim EMBED_DIM] [--hw-layers HW_LAYERS]
                [--hw-hsz HW_HSZ] [--latent-dim LATENT_DIM]
                [--dropout DROPOUT] [--enc-hsz ENC_HSZ]
                [--enc-layers ENC_LAYERS] [--dec-hsz DEC_HSZ]
                [--dec-layers DEC_LAYERS] [--clip CLIP]

GAN-NLG

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs for train
  --batch-size BATCH_SIZE
                        batch size for training
  --seed SEED           random seed
  --unuse-cuda          unuse cuda
  --n-warmup-steps N_WARMUP_STEPS
  --save SAVE           path to save the final model
  --data DATA           location of the data corpus
  --embed-dim EMBED_DIM
  --hw-layers HW_LAYERS
  --hw-hsz HW_HSZ
  --latent-dim LATENT_DIM
  --dropout DROPOUT
  --enc-hsz ENC_HSZ
  --enc-layers ENC_LAYERS
  --dec-hsz DEC_HSZ
  --dec-layers DEC_LAYERS
  --clip CLIP
```
