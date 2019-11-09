import os
import json

import argparse
import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.nn.functional import binary_cross_entropy

import data_loader
import const
import model
import utils

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str,
                    default=os.path.join(const.DATAPATH, "corpus.pt"))
parser.add_argument('--model_path', type=str, default='weights')
parser.add_argument('--cuda_device', type=str, default='0')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--max_len', type=int, default=500)
parser.add_argument('--span_len', type=int, default=4)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--pos_dim', type=int, default=20)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--rnn_hidden_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

use_cuda = torch.cuda.is_available()
corpus = torch.load(os.path.join(args.data))
args.word_ebd_weight = corpus["wordW"]
args.use_cuda = use_cuda

torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

corpus = torch.load(args.data)
args.wordW = corpus["wordW"]

train_and_test_data = data_loader.DataLoader(
    const.DATAPATH, corpus["word2idx"], cuda=use_cuda)

mention_pair_score = model.MentionPairScore(args)
if use_cuda:
    mention_pair_score = mention_pair_score.cuda()

optimizer = optim.Adam(mention_pair_score.parameters(), lr=args.learning_rate)


def train(i):
    mention_pair_score.train()
    total_loss = corrects = recall = ground_truth = 0
    for doc in tqdm(train_and_test_data.train_docs, mininterval=1, desc='pre-Train Processing', leave=False):
        optimizer.zero_grad()
        scores, labels = mention_pair_score(doc, corpus["word2idx"])
        loss = binary_cross_entropy(scores, labels, reduction='mean')
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
        predict = scores.gt(0.5).float()
        corrects += (predict*labels).sum().item()
        recall += predict.sum().item()
        ground_truth += labels.sum().item()

    f1 = 2*corrects/(recall+ground_truth)
    print(f"train epoch {i+1}/{args.epochs} loss: {total_loss/100:.4f} corrects: {corrects} recall: {recall} ground_truth: {ground_truth} f1: {f1:.4f}")


def dev(i):
    mention_pair_score.eval()
    total_loss = corrects = recall = ground_truth = 0

    for doc in tqdm(train_and_test_data.test_docs, mininterval=1, desc='pre-Dev Processing', leave=False):
        with torch.no_grad():
            scores, labels = mention_pair_score(doc, corpus["word2idx"])
            loss = binary_cross_entropy(scores, labels, reduction='mean')
            total_loss += loss.data.item()
            predict = scores.gt(0.5).float()
            corrects += (predict*labels).sum().item()
            recall += predict.sum().item()
            ground_truth += labels.sum().item()

    f1 = 2*corrects/(recall+ground_truth)
    print(f"dev epoch {i+1}/{args.epochs} loss: {total_loss/len(train_and_test_data.test_docs):.4f} corrects: {corrects} recall: {recall} ground_truth: {ground_truth} f1: {f1:.4f}")
    return f1


def save():
    model_state_dict = mention_pair_score.state_dict()
    model_source = {
        "settings": args,
        "model": model_state_dict,
        "word2idx": corpus['word2idx'],
    }
    torch.save(
        model_source, f"{os.path.join(args.model_path, 'pretrain_model.pt')}")


os.makedirs(args.model_path, exist_ok=True)
best_f1 = 0

try:
    print('-' * 90)
    for epoch in range(args.epochs):
        train(epoch)
        print('-' * 90)
        f1 = dev(epoch)
        print('-' * 90)
        if f1 >= best_f1:
            print(f"new best f1 score {f1:.4f} and save model")
            best_f1 = f1
            mention_pair_score.save_model(
                f"{os.path.join(args.model_path, 'middle_pretrain_model.pt')}")
            save()
        else:
            print(
                f"f1 score {f1:.4f} and reload best model best f1 {best_f1:.4f}")
            mention_pair_score.load_model(
                f"{os.path.join(args.model_path, 'middle_pretrain_model.pt')}", use_cuda)
        print('-' * 90)
except KeyboardInterrupt:
    print("Exiting from training early")
