## A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification
Module implemention from [paper](https://arxiv.org/pdf/1805.01089.pdf)

## Tutorial
Get [Tutorial](https://ne7ermore.github.io/post/hierarchical-sc/) if know Chinese

## Datasets - Amazon SNAP Review Dataset (SNAP)
We select three domains of product reviews to construct three benchmark datasets, which are Toys & Games, Sports & Outdoors, and Movie & TV.

Run in $(PROJECT_HOME) to download datasets
```
make download-data
```

## Requirement
* python 3.6.2
* pytorch 0.4.0
* numpy 1.14.0
* pandas 0.23.0

## How to train

### Prepare data
```
python3 fuel.py toys | sports | movies
python3 corpus.py
```

### train and predict
```
python3 train.py
```
