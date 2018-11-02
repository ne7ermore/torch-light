import torch

from const import *
from model import *


class Predict(object):
    def __init__(self, model_source, cuda=False):
        self.torch = torch.cuda if cuda else torch
        self.cuda = cuda
        if self.cuda:
            model_source = torch.load(model_source)
        else:
            model_source = torch.load(
                model_source, map_location=lambda storage, loc: storage)

        self.dict = model_source["word_dict"]
        self.label = model_source["label_dict"]
        self.args = args = model_source["settings"]

        model = DeepBiLSTMModel(args.word_size, args.label_size, args.word_ebd_dim,
                                args.lstm_hsz, args.lstm_layers, args.recurrent_dropout_prob, cuda)
        model.load_state_dict(model_source['model'])

        if self.cuda:
            model = model.cuda()
        else:
            model = model.cpu()

        self.model = model.eval()


if __name__ == "__main__":
    from data_loader import DataLoader

    p = Predict("weights/model_22.pt", True)

    data = torch.load("data/corpus.pt")
    validation_data = DataLoader(data['train']['word'],
                                 data['train']['label'],
                                 cuda=True,
                                 shuffle=True,
                                 batch_size=2)

    d = {v: k for k, v in data["dict"]["word"].items()}
    l = {v: k for k, v in data["dict"]["label"].items()}
    words, label = next(validation_data)
    pred = p.model(words)
    score, idxs = torch.max(pred, 1)

    for word in words.tolist():
        print(" ".join([d[w] for w in word]))
        print()

    idxs = idxs.view(2, -1)
    for idx in idxs.tolist():
        print(" ".join([l[_id] for _id in idx]))
        print()

    label = label.view(2, -1)
    for idx in label.tolist():
        print(" ".join([l[_id] for _id in idx]))
        print()
