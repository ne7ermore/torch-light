import torch
import torch.nn as nn
import torch.nn.functional as F

from const import *

from model import BERT


class Classifier(nn.Module):
    def __init__(self, lsz, args):
        super().__init__()

        self.bert = BERT(args)

        self.sent_predict = nn.Linear(args.d_model, lsz)
        self.sent_predict.weight.data.normal_(INIT_RANGE)
        self.sent_predict.bias.data.zero_()

    def get_trainable_parameters(self):
        return self.bert.get_trainable_parameters()

    def forward(self, inp, pos, segment_label):
        _, sent_encode = self.bert(inp, pos, segment_label)
        return F.log_softmax(self.sent_predict(sent_encode), dim=-1)

    def load_model(self, path="model.pt"):
        data = torch.load(path)
        self.bert.load_model(data["weights"])
