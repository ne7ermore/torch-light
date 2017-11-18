## Introduction
Natural Language Generation for Chinese Poetry

## Requirement
* python 3.5
* pytorch 0.2.0
* numpy 1.13.1
* tqdm

## Generator
```
>>> python3 generate.py

甚知无事事，不得访君还，山水春风起，江湖日夕曛。

鮌园不可见，独宿空山钟，不见春风起，相思不可寻。

价汪焜敷璧，玉赫奕羔斗，柄蹴鞠军门，雄战马行兵。

荪渚草草绿，春园春草生，春风吹柳树，春色满山川。

韮糵榱弧纛，龙旂拂昊虹，霓裳凌紫塞，鹰隼集羣公。

醅香花落落，日落月明明，月照山光晚，光凝晓色寒。
```


## Train
```
python3 train.py -h
```

You will get:

```
usage: train.py [-h] [--lr LR] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--seed SEED] [--cuda-able] [--save SAVE] [--data DATA]
                [--dropout DROPOUT] [--embed-dim EMBED_DIM]
                [--hidden-size HIDDEN_SIZE] [--lstm-layers LSTM_LAYERS]
                [--clip CLIP] [--bidirectional]

NLG for Chinese Poetry

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               initial learning rate
  --epochs EPOCHS       number of epochs for train
  --batch-size BATCH_SIZE
                        batch size for training
  --seed SEED           random seed
  --cuda-able           enables cuda
  --save SAVE           path to save the final model
  --data DATA           location of the data corpus
  --dropout DROPOUT     the probability for dropout (0 = no dropout)
  --embed-dim EMBED_DIM
                        number of embedding dimension
  --hidden-size HIDDEN_SIZE
                        number of lstm hidden dimension
  --lstm-layers LSTM_LAYERS
                        biLSTM layer numbers
  --clip CLIP           gradient clipping
  --bidirectional       If True, becomes a bidirectional LSTM
```
