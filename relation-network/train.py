import argparse
import os

import torch

from data_loader import DataLoader
from model import *
from const import PAD

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=1101)
parser.add_argument('--unuse_cuda', action='store_true')

parser.add_argument('--data', type=str, default='data/corpus.pt')

parser.add_argument('--word_ebd_dim', type=int, default=32)
parser.add_argument('--lstm_hsz', type=int, default=64)
parser.add_argument('--lstm_layers', type=int, default=2)
parser.add_argument('--clip', type=float, default=5.)
parser.add_argument('--lr', type=float, default=2e-4)

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

training_data = DataLoader(data["story"],
                           data["question"],
                           data["answer"],
                           data["max_q_len"],
                           data["max_s_len"],
                           data["word2idx"],
                           data["answer2idx"],
                           cuda=use_cuda,
                           batch_size=args.batch_size)

validation_data = DataLoader(data["test_story"],
                             data["test_question"],
                             data["test_answer"],
                             data["max_q_len"],
                             data["max_s_len"],
                             data["word2idx"],
                             data["answer2idx"],
                             cuda=use_cuda,
                             batch_size=args.batch_size,
                             shuffle=False)

model = RelationNet(len(data["word2idx"]),
                    len(data["answer2idx"]),
                    data["max_s_len"],
                    data["max_q_len"],
                    use_cuda,
                    story_hsz=args.lstm_hsz,
                    story_layers=args.lstm_layers,
                    question_hsz=args.lstm_hsz,
                    question_layers=args.lstm_layers)

if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()


def train(i):
    model.train()
    corrects = total_loss = 0
    for story, question, answer in training_data:
        optimizer.zero_grad()
        pred = model(story, question)
        loss = criterion(pred, answer)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.data.item()
        corrects += (torch.max(pred, 1)[1].data == answer.data).sum()
    print(
        f"train epoch {i+1}/{args.epochs} loss: {total_loss/training_data.stop_step:.4f} corrects: {float(corrects)*100/(training_data.sents_size):.2f}%")

    if tf is not None:
        add_summary_value("train loss", total_loss /
                          training_data.stop_step, i)
        add_summary_value("train corrects", float(corrects)
                          * 100/(training_data.sents_size), i)
        tf_summary_writer.flush()


def evaluate(i):
    model.eval()
    corrects = eval_loss = 0

    for story, question, answer in validation_data:
        with torch.no_grad():
            pred = model(story, question)
            loss = criterion(pred, answer)
            eval_loss += loss.data.item()
            corrects += (torch.max(pred, 1)[1].data == answer.data).sum()
    print(
        f"eval epoch {i+1}/{args.epochs} loss: {eval_loss/validation_data.stop_step:.4f} corrects: {float(corrects)*100/(validation_data.sents_size):.2f}%")

    if tf is not None:
        add_summary_value("evaluate loss", eval_loss /
                          validation_data.stop_step, i)
        add_summary_value("evaluate corrects", float(corrects)
                          * 100 / (validation_data.sents_size), i)
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
            "word2idx": data['word2idx'],
            "answer2idx": data['answer2idx'],
            "max_q_len": data["max_q_len"],
            "max_s_len": data["max_s_len"],
        }
        torch.save(model_source, f"weights/model_{epoch+1}.pt")

except KeyboardInterrupt:
    print("Exiting from training early")
