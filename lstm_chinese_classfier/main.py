import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init

import numpy as np
useCuda = torch.cuda.is_available()
torch.cuda.manual_seed(1) if useCuda else torch.manual_seed(1)

train_data = (("这件 衣服 多少 钱", "价格"),
    ("老板 便宜 点 好吗", "议价"),
    ("快递 好久 到", "快递"),
    ("裤子 好久 能 送到", "快递"),
    ("老板 在吗", "起始语"),
    ("裤子 多少 钱 啊 老板", "价格"))
# mini batch
bsz = 2
# sentence fixed length
embd_length = 10
# word for filling
EOS = "e"
learn_rate = 0.001
epoch = 50

class LSTModule(nn.Module):
    def __init__(self, tokens, label_size, embd_length, ebd_dim=64, hidden_size=128, n_layers=3, dropout=0.5, batch_first=True):
        super().__init__()
        self.embd_length = embd_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.ebd = nn.Embedding(tokens, ebd_dim)
        self.lstm = nn.LSTM(ebd_dim, hidden_size, n_layers, dropout=dropout, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size*embd_length, label_size)
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights.
        Avoid all datas from zero
        """
        self.ebd.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.n_layers, self.hidden_size)),
                            Variable(torch.zeros(batch_size, self.n_layers, self.hidden_size)))

    def forward(self, input, hidden):
        encode = self.ebd(input)
        output, hidden = self.lstm(encode, hidden)
        out = self.linear(output.contiguous().view(output.size(0), -1))
        return out

def parse_data(data):
    corpus_dict = {EOS: 0}
    label_dic = {}
    n_corpus = 1
    n_labels = 0
    for d in data:
        for corpus in d[0].split():
            if corpus not in corpus_dict.keys():
                corpus_dict[corpus] = n_corpus
                n_corpus += 1

        if d[1] not in label_dic.keys():
            label_dic[d[1]] = n_labels
            n_labels += 1

    return corpus_dict, label_dic

corpus_dict, label_dic = parse_data(train_data)

def sentences_to_tensor(sentences):
    sentences_len = (len(sentences) // bsz) * bsz # discard unused sentences
    data = np.zeros((sentences_len, embd_length))
    label = np.zeros(sentences_len)
    for i in range(0, sentences_len):
        label[i] = label_dic[sentences[i][1]]

        l = sentences[i][0].split()
        if len(l) > embd_length:
            l = l[:embd_length]

        for j, corpus in enumerate(l):
            data[i][j] = corpus_dict[corpus]
    return torch.from_numpy(data).type(torch.LongTensor), torch.from_numpy(label).type(torch.LongTensor)

def get_batch(train, target, i):
    if useCuda:
        return Variable(train[i:i+bsz]).cuda(), Variable(target[i:i+bsz]).cuda()
    return Variable(train[i:i+bsz]), Variable(target[i:i+bsz])

model = LSTModule(len(corpus_dict), len(label_dic), embd_length)
if useCuda:
    model.cuda()

def parse_hidden(hidden):
    if isinstance(hidden, Variable):
        return hidden.cuda()
    else:
        return tuple(parse_hidden(v) for v in hidden)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

train, target = sentences_to_tensor(train_data)
for e in range(1, epoch+1):
    model.train()
    print("epoch - [{}]".format(e))
    for batch, i in enumerate(range(0, len(train_data)-1, bsz)):
        hidden = model.init_hidden(bsz)
        if useCuda:
            hidden = parse_hidden(hidden)

        # Step 1. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        _train, _target = get_batch(train, target, i)

        # Step 2. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        optimizer.zero_grad()

        # Step 3. Run our forward pass.
        output = model(_train, hidden)

        # Step 4. Compute the loss, gradients, and update the parameters
        loss = criterion(output, _target)
        loss.backward()
        optimizer.step()
        print("loss - [{}]".format((loss.data)[0]))
