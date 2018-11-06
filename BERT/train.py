import random

import torch
from torch.utils.data import DataLoader
import numpy as np

from model import BERT, WordCrossEntropy, ScheduledOptim
from data_loader import BERTDataSet
import const


def train(args):
    assert torch.cuda.is_available(), "need to use GPUs"

    use_cuda = torch.cuda.is_available()
    cuda_devices = list(map(int, args.cuda_devices.split(",")))
    is_multigpu = len(cuda_devices) > 1
    device = "cuda"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if is_multigpu > 1:
        torch.cuda.manual_seed_all(args.seed)

    data = torch.load(args.data)
    dataset = BERTDataSet(data['word'],
                          data['max_len'],
                          data["dict"],
                          args.batch_size * args.steps)
    training_data = DataLoader(dataset,
                               batch_size=args.batch_size,
                               num_workers=args.num_cpus)

    model = BERT(dataset.word_size,
                 data["max_len"],
                 args.n_stack_layers,
                 args.d_model,
                 args.d_ff,
                 args.n_head,
                 args.dropout)

    optimizer = ScheduledOptim(
        nn.DataParallel(torch.optim.Adam(
            model.get_trainable_parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-09,
            weight_decay=0.01),
            device_ids=cuda_devices),
        args.d_model,
        args.n_warmup_steps)

    w_criterion = WordCrossEntropy()
    s_criterion = torch.nn.CrossEntropyLoss()

    if use_cuda:
        model = model.to(device)
        w_criterion = w_criterion.to(device)

    model = torch.nn.DataParallel(model, device_ids=cuda_devices)
    model.train()
    for step, datas in enumerate(training_data):
        inp, pos, sent_label, word_label, segment_label = list(
            map(lambda x: x.to(device), datas))
        optimizer.zero_grad()
        word, sent = model(inp, pos, segment_label)
        w_loss, w_corrects, tgt_sum = w_criterion(word, word_label)
        s_loss = s_criterion(sent, sent_label)
        if is_multigpu:
            w_loss, s_loss = w_loss.mean(), s_loss.mean()
        loss = w_loss + s_loss
        loss.backward()
        optimizer.step()
        s_corrects = (torch.max(sent, 1)[1].data == sent_label.data).sum()

        print(f"[Step {step+1}/{args.steps}] [word_loss: {w_loss:.5f}, sent_loss: {s_loss:.5f}, loss: {loss:.5f}, word_corrects: {w_corrects/tgt_sum*100}%], sent_corrects: {s_corrects/args.batch_size*100}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=601)
    parser.add_argument('--num_cpus', type=int, default=6)
    parser.add_argument('--cuda_devices', type=str, default='0,1,2')
    parser.add_argument('--save', type=str, default='bert.pt')
    parser.add_argument('--data', type=str, default='data/corpus.pt')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_stack_layers', type=int, default=12)
    parser.add_argument('--n_warmup_steps', type=int, default=10000)

    args = parser.parse_args()

    train(args)
