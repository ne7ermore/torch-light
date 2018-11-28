import random

import torch
from torch.utils.data import DataLoader
import numpy as np

from model import WordCrossEntropy
from optimization import AdamWeightDecayOptimizer
from pretrain import Pretraining
from data_loader import BERTDataSet


try:
    import tensorflow as tf
except ImportError:
    tf = None

tf_summary_writer = tf and tf.summary.FileWriter("logdir")


def add_summary_value(key, value, tf_step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    tf_summary_writer.add_summary(summary, tf_step)


def main(args):
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

    args.max_len = data["max_len"]
    args.vsz = dataset.word_size
    model = Pretraining(2, args)

    print(
        f"BERT have {model.bert.parameters_count()} paramerters in total")

    optimizer = AdamWeightDecayOptimizer(
        model.get_optimizer_parameters(args.weight_decay),
        lr=args.lr,
        warmup=args.n_warmup_steps,
        train_steps=args.steps,
        weight_decay=args.weight_decay,
        clip=args.clip
    )

    w_criterion = WordCrossEntropy()
    w_criterion = w_criterion.to(device)

    s_criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()
    paral_model = torch.nn.DataParallel(model, device_ids=cuda_devices)

    for step, datas in enumerate(training_data):
        inp, pos, sent_label, word_label, segment_label = list(
            map(lambda x: x.to(device), datas))
        sent_label = sent_label.view(-1)
        optimizer.zero_grad()
        word, sent = paral_model(inp, pos, segment_label)
        w_loss, w_corrects, tgt_sum = w_criterion(word, word_label)
        s_loss = s_criterion(sent, sent_label)
        loss = w_loss + s_loss
        loss.backward()
        optimizer.step()
        s_corrects = (torch.max(sent, 1)[1].data == sent_label.data).sum()

        print(f"[Step {step+1}/{args.steps}] [word_loss: {w_loss:.5f}, sent_loss: {s_loss:.5f}, loss: {loss:.5f}, w_pre: {w_corrects/tgt_sum*100:.2f}% {w_corrects}/{tgt_sum}, s_pre: {float(s_corrects)/args.batch_size*100:.2f}% {s_corrects}/{args.batch_size}]")

        if tf is not None:
            add_summary_value("Word loss", w_loss, step)
            add_summary_value("Sent loss", s_loss, step)
            add_summary_value("Loss", loss, step)
            add_summary_value("Word predict", w_corrects / tgt_sum, step)
            add_summary_value("Sent predict", float(
                s_corrects) / args.batch_size, step)
            tf_summary_writer.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_cpus', type=int, default=5)
    parser.add_argument('--cuda_devices', type=str, default='0,6,7')
    parser.add_argument('--save', type=str, default='bert.pt')
    parser.add_argument('--data', type=str, default='data/corpus.pt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_stack_layers', type=int, default=12)
    parser.add_argument('--n_warmup_steps', type=int, default=10000)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=.01)

    args = parser.parse_args()

    main(args)
