import os
import json

import argparse
import torch
from tqdm import tqdm
from torch.nn.functional import binary_cross_entropy

import data_loader
import const
import model
import common
import predict

parser = argparse.ArgumentParser()

parser.add_argument('--unuse_cuda', action='store_true')
parser.add_argument('--data', type=str, default="data/corpus.pt")
parser.add_argument('--model_path', type=str, default='weights')
parser.add_argument('--cuda_device', type=str, default='0')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--warm_epochs', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1111)

parser.add_argument('--dilation_rates', type=str,
                    default='1,2,5,1,2,5,1,2,5,1,1,1')
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--n_warmup_steps', type=int, default=0)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

use_cuda = torch.cuda.is_available() and not args.unuse_cuda

torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

data = torch.load(args.data)
if "charW" in data and "wordW" in data:
    args.charW, args.wordW = data["charW"], data["wordW"]

training_data = data_loader.DataLoader(data["train"]["char"],
                                       data["train"]["word"],
                                       data["train"]["sub_sidx"],
                                       data["train"]["sub_eidx"],
                                       data["train"]["obj_idxs"],
                                       data["train"]["sub_slidx"],
                                       data["train"]["sub_elidx"],
                                       data["word2idx"],
                                       data["char2idx"],
                                       data["predicate2id"],
                                       cuda=use_cuda,
                                       batch_size=args.batch_size)

validation_data = data_loader.DataLoader(data["dev"]["char"],
                                         data["dev"]["word"],
                                         data["dev"]["sub_sidx"],
                                         data["dev"]["sub_eidx"],
                                         data["dev"]["obj_idxs"],
                                         data["dev"]["sub_slidx"],
                                         data["dev"]["sub_elidx"],
                                         data["word2idx"],
                                         data["char2idx"],
                                         data["predicate2id"],
                                         cuda=use_cuda,
                                         batch_size=args.batch_size,
                                         shuffle=False)

args.n_warmup_steps = args.n_warmup_steps if args.n_warmup_steps and args.n_warmup_steps != 0 else training_data.stop_step
args.char_size = len(data["char2idx"])
args.word_size = len(data["word2idx"])
args.num_classes = len(data["predicate2id"])
args.dilation_rates = list(map(int, args.dilation_rates.split(",")))
args.max_len = data["max_len"]
args.inp_dim = 200

model = model.Model(args)
if use_cuda:
    model = model.cuda()

optimizer = common.ScheduledOptim(
    torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    args.d_model, args.n_warmup_steps)


def mask_binary_cross_entropy(inp, target, mask):
    loss = binary_cross_entropy(inp, target, reduction='none')
    return torch.sum(loss*mask) / torch.sum(mask)


def dev(i):
    model.eval()
    total_loss = 0
    for chars, words, position, sub_sidx, sub_eidx, obj_sidx, obj_eidx, sub_slidx, sub_elidx in tqdm(validation_data, mininterval=1, desc='dev Processing', leave=False):
        with torch.no_grad():
            p_sub_sidx, p_sub_eidx, p_obj_sidx, p_obj_eidx, mask = model(
                chars, words, position, sub_slidx, sub_elidx)

            ss_loss = mask_binary_cross_entropy(p_sub_sidx, sub_sidx, mask)
            se_loss = mask_binary_cross_entropy(p_sub_eidx, sub_eidx, mask)
            os_loss = mask_binary_cross_entropy(p_obj_sidx, obj_sidx, mask)
            oe_loss = mask_binary_cross_entropy(p_obj_eidx, obj_eidx, mask)

            loss = ss_loss+se_loss+os_loss+oe_loss
            total_loss += loss.data.item()

    print(
        f"dev epoch {i+1}/{args.epochs} loss: {total_loss/training_data.stop_step:.4f}")


def train(i):
    model.train()
    total_loss = 0
    for chars, words, position, sub_sidx, sub_eidx, obj_sidx, obj_eidx, sub_slidx, sub_elidx in tqdm(training_data, mininterval=1, desc='Train Processing', leave=False):
        optimizer.zero_grad()
        p_sub_sidx, p_sub_eidx, p_obj_sidx, p_obj_eidx, mask = model(
            chars, words, position, sub_slidx, sub_elidx)

        ss_loss = mask_binary_cross_entropy(p_sub_sidx, sub_sidx, mask)
        se_loss = mask_binary_cross_entropy(p_sub_eidx, sub_eidx, mask)
        os_loss = mask_binary_cross_entropy(p_obj_sidx, obj_sidx, mask)
        oe_loss = mask_binary_cross_entropy(p_obj_eidx, obj_eidx, mask)

        loss = ss_loss+se_loss+os_loss+oe_loss
        loss.backward()

        optimizer.step()
        optimizer.update_learning_rate()
        total_loss += loss.data.item()

    print(
        f"train epoch {i+1}/{args.epochs} loss: {total_loss/training_data.stop_step:.4f}")


def test(i, predict):
    model.eval()
    t = pre = groud = 0
    inf = open("data/dev_data.json", encoding="utf8")
    for line in inf:
        line = json.loads(line)
        text = line["text"]
        g_triples = set()
        for trip in line["spo_list"]:
            g_triples.add((trip["subject"], trip["predicate"], trip["object"]))

        p_triples = predict.predict(text)
        pre += len(p_triples)
        groud += len(g_triples)
        t += len(p_triples.intersection(g_triples))

    print(
        f"test epoch {i+1}/{args.epochs} precision: {t/(pre+0.001):.4f} recall: {t/groud:.4f} f1: {2*t/(pre+groud):.4f}")
    return 2*t/(pre+groud)


def save():
    model_state_dict = model.state_dict()
    model_source = {
        "settings": args,
        "model": model_state_dict,
        "word2idx": data['word2idx'],
        "char2idx": data['char2idx'],
        "max_len": data["max_len"],
        "predicate2id": data["predicate2id"],
    }
    torch.save(model_source, f"{os.path.join(args.model_path, 'model.pt')}")


os.makedirs("weights", exist_ok=True)

data["model"] = model
predict = predict.Predict(model_datas=data, cuda=True)
best_f1 = 0

try:
    print('-' * 90)
    for epoch in range(args.epochs):
        train(epoch)
        print('-' * 90)
        dev(epoch)
        print('-' * 90)
        predict.update_model(model)
        f1 = test(epoch, predict)
        if f1 > best_f1:
            print(f"new best f1 score {f1:.4f} and save model")
            best_f1 = f1
            model.save_model(
                f"{os.path.join(args.model_path, 'tmp_model.pt')}")
            save()
        else:
            print(f"f1 {f1} and reload best model")
            model.load_model(
                f"{os.path.join(args.model_path, 'tmp_model.pt')}", use_cuda)
        print('-' * 90)
except KeyboardInterrupt:
    print("Exiting from training early")
