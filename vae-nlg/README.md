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
鱼涛迷山岛，浩去一萧然，相访陶陵郡，长来道姓稀。

大县饶孤渚，乡人夜月深，虽闻三岳外，独入一陵人。

大公皆在郡，汉陌即同贤，莫覩周陵秀，天涯御苑心。

三峰有片下，大业在樵卿，仲水多萧漠，湲阖接九溟。

寓者望此海，征溟叠苍漠，漠渚连湘寺，斜吟枕寺枝。

山颇无风如，光骸其混肌，义肃氛旗凝，云旗苍崖涧。

虎骨成龙穴，防风拂白霞，竹花秋作散，芳径落红荪。

病眠闲涧径，唯得怯猨痕，大海长流阔，湘帆次北城。
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
