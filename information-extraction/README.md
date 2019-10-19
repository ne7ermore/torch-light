## A module for information extraction

## Requirement
* python 3.7
* pytorch 1.0.0
* numpy 1.13.1

## Usage

### Predict

```
python3 predict.py

《李白》是李荣浩作词作曲并演唱的歌曲，该曲收录于2013年9月17号发行的原创专辑《模特》中
{('李白', '作词', '李荣浩'), ('李白', '作曲', '李荣浩'), ('李白', '歌手', '李荣浩'), ('李白', '所属专辑', '模特')}

```

### train

Step.1 - prepare corpus

```python
python3 corpus.py
```

Step.2 - Train data
```python
python3 train.py
```

## DataBase

download data
```
python3 download.py
```