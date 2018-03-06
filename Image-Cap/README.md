## Introduction
Image Captioning.
Module implemention from "[Actor-Critic Sequence Training for Image Captioning](https://arxiv.org/abs/1706.09601)". <br>

## Requirement
* python 3.5
* pytorch 0.3.1
* numpy 1.13.1
* PIL

## Dataset - COCO
Download the coco datas from [here](http://cocodataset.org/#download). We use train2017 and val2017

## Model training
* Step 1 - Download dataset, move train2017/ val2017/ captions_train2017.json and captions_val2017.json to {MODEL_HOME}/data/ <br>

* Step 2 - Run caption.py
```
$ python3 caption.py
```
* Step 3 - Train model
```
$ python3 train.py
```

## Result

<p align="center">My Captioning: Vase with a flower in it sitting on a table.</p>
<p align="center"><img src="result/000000249025.jpg" /></p>

***

<p align="center">My Captioning: Cat is laying on a couch with a stuffed animal.</p>
<p align="center"><img src="result/000000046378.jpg" /></p>

***

<p align="center">My Captioning: City street filled with lots of tall buildings.</p>
<p align="center"><img src="result/000000221754.jpg" /></p>

***

<p align="center">My Captioning: Desk with a computer and a laptop on it.</p>
<p align="center"><img src="result/000000063740.jpg" /></p>

***

<p align="center">My Captioning: Man wearing a black jacket and a white shirt.</p>
<p align="center"><img src="result/000000228214.jpg" /></p>