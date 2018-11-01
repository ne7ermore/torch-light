import argparse
import os

import torch

from data_loader import DataLoader
from model import *
from const import PAD

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--seed', type=int, default=1101)
parser.add_argument('--unuse_cuda', action='store_true')

parser.add_argument('--data', type=str, default='data/corpus.pt')

parser.add_argument('--word_ebd_dim', type=int, default=128)
parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1)
parser.add_argument('--lstm_hsz', type=int, default=256)
parser.add_argument('--lstm_layers', type=int, default=8)
parser.add_argument('--clip', type=float, default=1.)

args = parser.parse_args()

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and not args.unuse_cuda

if use_cuda:
    torch.cuda.manual_seed(args.seed)


# Tensorboard
try:
    import tensorflow as tf
except ImportError:
    tf = None

tf_summary_writer = tf and tf.summary.FileWriter("logdir")


def add_summary_value(key, value, tf_step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    tf_summary_writer.add_summary(summary, tf_step)


data = torch.load(args.data)
args.word_max_len = data["word_max_len"]
args.word_size = data['dict']['word_size']
args.label_size = data['dict']['label_size']

training_data = DataLoader(
    data['train']['word'],
    data['train']['label'],
    cuda=use_cuda,
    batch_size=args.batch_size)

validation_data = DataLoader(
    data['valid']['word'],
    data['valid']['label'],
    batch_size=args.batch_size,
    shuffle=False,
    cuda=use_cuda)

model = DeepBiLSTMModel(args.word_size, args.label_size, args.word_ebd_dim,
                        args.lstm_hsz, args.lstm_layers, args.recurrent_dropout_prob, use_cuda)

if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adadelta(model.parameters(), rho=.9)
criterion = torch.nn.CrossEntropyLoss()


def train(i):
    model.train()
    sums = corrects = total_loss = 0
    for word, label in training_data:
        optimizer.zero_grad()
        pred = model(word)
        loss = criterion(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.data.item()
        corrects += (torch.max(pred, 1)[1].data == label.data).sum()
        sums += label.shape[0]
    print(
        f"train epoch {i+1}/{args.epochs} loss: {total_loss/training_data._stop_step:.4f} corrects: {float(corrects)*100/(sums):.2f}%")

    if tf is not None:
        add_summary_value("train loss", total_loss /
                          training_data._stop_step, i)
        add_summary_value("train corrects", float(corrects) * 100 / (sums), i)
        tf_summary_writer.flush()


def evaluate(i):
    model.eval()
    sums = corrects = eval_loss = 0

    for word, label in validation_data:
        with torch.no_grad():
            pred = model(word)
            loss = criterion(pred, label)
            eval_loss += loss.data.item()
            corrects += (torch.max(pred, 1)[1].data == label.data).sum()
            sums += label.shape[0]

    print(
        f"eval epoch {i+1}/{args.epochs} loss: {eval_loss/validation_data._stop_step:.4f} corrects: {float(corrects)*100/(sums):.2f}%")

    if tf is not None:
        add_summary_value("evaluate loss", eval_loss /
                          validation_data._stop_step, i)
        add_summary_value("evaluate corrects", float(
            corrects) * 100 / (sums), i)
        tf_summary_writer.flush()


os.makedirs("weights", exist_ok=True)
try:
    print('-' * 90)
    for epoch in range(args.epochs):
        train(epoch)
        print('-' * 90)
        evaluate(epoch)
        print('-' * 90)

        model_state_dict = model.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict,
            "word_dict": data['dict']['word'],
            "label_dict": data['dict']['label']
        }
        torch.save(model_source, f"weights/model_{epoch+1}.pt")

except KeyboardInterrupt:
    print("Exiting from training early")
