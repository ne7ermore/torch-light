## A simple neural network module for relational reasoning
Module implemention from "[A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427)" 

In this paper we describe how to use Relation Networks (RNs) as a simple plug-and-play module to solve problems that fundamentally hinge on relational reasoning.

## Tutorial
Get [Tutorial](https://ne7ermore.github.io/post/relation-net/) if know Chinese


## Requirement
* python 3.7
* pytorch 1.0.0
* numpy 1.13.1

## DataBase
bAbI - bAbI is a pure text-based QA dataset. There are 20 tasks, each corresponding to a particular type of reasoning, such as deduction, induction, or counting.

## train

Step.1 - prepare corpus

```python
python3 corpus.py
```

Step.2 - Train data
```python
python3 train.py
```