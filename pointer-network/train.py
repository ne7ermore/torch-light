import argparse
import os

import torch
from tqdm import tqdm

from data_loader import DataLoader
import const
from model import Transformer, CrossEntropy

parser = argparse.ArgumentParser(description='poiter network')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--cuda_device', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--model_path', type=str, default='weights')
parser.add_argument('--data', type=str, default=f'./data/corpus.pt')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--d_k', type=int, default=64)
parser.add_argument('--d_v', type=int, default=64)
parser.add_argument('--n_stack_layers', type=int, default=6)
parser.add_argument('--learning_rate', type=float, default=0.00005)

args = parser.parse_args()
args.turn_size = 4
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(args.seed)

corpus = torch.load(args.data)
args.vocab_size = len(corpus["word2idx"])
args.max_context_len = corpus["max_len"]

training_data = DataLoader(
    corpus["train"]["src_texts"],
    corpus["train"]["src_turn"],
    corpus["train"]["tgt_indexs"],
    corpus["train"]["tgt_texts"],
    batch_size=args.batch_size,
    cuda=use_cuda)

validation_data = DataLoader(
    corpus["valid"]["src_texts"],
    corpus["valid"]["src_turn"],
    corpus["valid"]["tgt_indexs"],
    corpus["valid"]["tgt_texts"],
    batch_size=args.batch_size,
    cuda=use_cuda)

model = Transformer(args)

criterion = CrossEntropy()
optimizer = torch.optim.Adam(
    model.get_trainable_parameters(), lr=args.learning_rate)

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()


def get_performance(crit, distributes, gold):
    loss = crit(distributes, gold)
    _, predict = distributes.max(dim=-1)
    n_correct = predict.eq(gold)
    n_correct = n_correct.data.masked_select(gold.ne(const.PAD)).sum()

    n_gold = gold.ne(const.PAD).sum()

    return loss, n_correct, n_gold


def dev(i):
    model.eval()
    total_loss = total_correct = total_gold = 0
    for (src_tensor, src_postion, turns_tensor), (tgt_tensor, tgt_postion), tgt_indexs_tensor in tqdm(validation_data, mininterval=1, desc='dev Processing', leave=False):
        with torch.no_grad():
            distributes = model(src_tensor, src_postion,
                                turns_tensor, tgt_tensor)

            loss, n_correct, n_gold = get_performance(
                criterion, distributes, tgt_indexs_tensor)
            total_loss += loss.item()
            total_correct += n_correct.item()
            total_gold += n_gold.item()

    print(f"dev epoch {i+1}/{args.epochs} loss: {total_loss/validation_data.stop_step:.4f} correct: {total_correct} gold count: {total_gold} presicion: {total_correct/total_gold:.4f}")
    return total_correct/total_gold


def train(i):
    model.train()
    total_loss = total_correct = total_gold = 0
    for (src_tensor, src_postion, turns_tensor), (tgt_tensor, tgt_postion), tgt_indexs_tensor in tqdm(training_data, mininterval=1, desc='Train Processing', leave=False):
        optimizer.zero_grad()
        distributes = model(src_tensor, src_postion, turns_tensor, tgt_tensor)

        loss, n_correct, n_gold = get_performance(
            criterion, distributes, tgt_indexs_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += n_correct.item()
        total_gold += n_gold.item()

    print(f"train epoch {i+1}/{args.epochs} loss: {total_loss/training_data.stop_step:.4f} correct: {total_correct} gold count: {total_gold} presicion: {total_correct/total_gold:.4f}")


def save():
    model_state_dict = model.state_dict()
    model_source = {
        "settings": args,
        "model": model_state_dict,
        "word2idx": corpus['word2idx'],
    }
    torch.save(model_source, f"{args.model_path}/model_{args.cuda_device}.pt")


os.makedirs(args.model_path, exist_ok=True)


best_presicion = -1

try:
    print('-' * 90)
    for epoch in range(args.epochs):
        train(epoch)
        print('-' * 90)
        presicion = dev(epoch)
        if presicion > best_presicion:
            print(f"new best presicion score {presicion:.4f} and save model")
            best_presicion = presicion
            model.save_model(
                f"{args.model_path}/tmp_model_{args.cuda_device}.pt")
            save()
        else:
            print(f"best_presicion {best_presicion:.4f} and reload best model")
            model.load_model(
                f"{args.model_path}/tmp_model_{args.cuda_device}.pt", use_cuda)
        print('-' * 90)
except KeyboardInterrupt:
    print("Exiting from training early")
