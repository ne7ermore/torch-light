import torch.nn as nn
import torch.nn.functional as F

from const import *

from model import BERT


class Pretraining(nn.Module):
    def __init__(self, lsz, args):
        super().__init__()

        self.bert = BERT(args)

        self.sent_predict = nn.Linear(args.d_model, lsz)
        self.word_predict = nn.Linear(args.d_model, args.vsz)

        self.reset_parameters()

    def reset_parameters(self):
        self.bert.reset_parameters()

        self.sent_predict.weight.data.normal_(INIT_RANGE)
        self.sent_predict.bias.data.zero_()

        self.word_predict.weight = self.bert.enc_ebd.weight  # share weights
        self.word_predict.bias.data.zero_()

    def get_optimizer_parameters(self, decay):
        return [{'params': [p for n, p in self.named_parameters(
        ) if n.split(".")[-1] not in NOT_USE_WEIGHT_DECAY and p.requires_grad], 'weight_decay': decay},
            {'params': [p for n, p in self.named_parameters(
            ) if n.split(".")[-1] in NOT_USE_WEIGHT_DECAY and p.requires_grad], 'weight_decay': 0.0}]

    def forward(self, inp, pos, segment_label):
        word_encode, sent_encode = self.bert(inp, pos, segment_label)

        sent = F.log_softmax(self.sent_predict(sent_encode), dim=-1)
        word = F.log_softmax(self.word_predict(word_encode), dim=-1)

        return word, sent


if __name__ == "__main__":
    import torch
    import data_loader
    from torch.utils.data import DataLoader
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_cpus', type=int, default=5)
    parser.add_argument('--cuda_devices', type=str, default='0,3,5,6,7')
    parser.add_argument('--save', type=str, default='bert.pt')
    parser.add_argument('--data', type=str, default='data/corpus.pt')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_stack_layers', type=int, default=12)
    parser.add_argument('--n_warmup_steps', type=int, default=10000)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    args = parser.parse_args()

    data = torch.load("data/corpus.pt")
    ds = data_loader.BERTDataSet(
        data["word"], data["max_len"], data["dict"], 10000)
    train_data_loader = DataLoader(ds, batch_size=3, num_workers=5)
    s_criterion = torch.nn.CrossEntropyLoss()
    device_ids = [0, 2]
    args.max_len = data["max_len"]
    args.vsz = ds.word_size
    b = Pretraining(2, args)
    b = b.cuda(device_ids[0])
    paral_b = torch.nn.DataParallel(b, device_ids=device_ids)
    print(
        f"BERT have {b.bert.parameters_count()} paramerters in total")
    for datas in train_data_loader:
        inp, pos, sent_label, word_label, segment_label = list(
            map(lambda x: x.cuda(device_ids[0]), datas))

        word, sent = paral_b(inp, pos, segment_label)
        print(word.shape)
        print(sent.shape)
        print(torch.max(sent, 1)[1])
        print(sent_label.shape)

        s_criterion(sent, sent_label.view(-1))
        b.bert.save_model(args, data)

        time.sleep(2)
