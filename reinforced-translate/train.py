import argparse

parser = argparse.ArgumentParser(
    description='A DEEP REINFORCED MODEL translate')

parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--save', type=str, default='translate_{}.pt')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--pretrain_epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--data', type=str, default='./data/corpus')
parser.add_argument('--ml_lr', type=float, default=0.001)
parser.add_argument('--rl_lr', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--emb_dim', type=int, default=100)
parser.add_argument('--rnn_hsz', type=int, default=100)
parser.add_argument('--pretrain', type=str, default="")

args = parser.parse_args()

import torch

torch.manual_seed(args.seed)
args.use_cuda = use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.cuda.manual_seed(args.seed)

# ##############################################################################
# Tensorboard
################################################################################
try:
    import tensorflow as tf
    tf_step = 0
except ImportError:
    tf = None

tf_summary_writer = tf and tf.summary.FileWriter(args.logdir)


def add_summary_value(key, value):
    global tf_step

    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    tf_summary_writer.add_summary(summary, tf_step)


# ##############################################################################
# Load data
################################################################################
import time

from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_len"]
args.src_vs = data['dict']['src_size']
args.tgt_vs = data['dict']['tgt_size']

training_data = DataLoader(
    data['train']['data'],
    data['train']['label'],
    data['max_len'],
    use_cuda,
    batch_size=args.batch_size)

validation_data = DataLoader(
    data['valid']['data'],
    data['valid']['label'],
    data['max_len'],
    use_cuda,
    batch_size=args.batch_size)

args.src_id2w = data['dict']['src_id2w']
args.tgt_id2w = data['dict']['tgt_id2w']

# ##############################################################################
# Training
# ##############################################################################
from const import PAD
from model import Model, CrossEntropy, SelfCriticCriterion

model = Model(args)
ml_optimizer = torch.optim.Adam(model.parameters(), lr=args.ml_lr)
rl_optimizer = torch.optim.Adam(
    model.parameters(), lr=args.rl_lr, weight_decay=0.01)

ml_criterion = CrossEntropy()
rl_criterion = SelfCriticCriterion()

if use_cuda:
    model = model.cuda()

# ##############################################################################
# Training
# ##############################################################################
from tqdm import tqdm

from torch.autograd import Variable
import torch.nn.functional as F

from rouge import rouge_l, mask_score


def pre_train():
    if tf:
        global tf_step
    for src, tgt in tqdm(training_data,
                         mininterval=1,
                         desc="Pre-train",
                         leave=False):
        ml_optimizer.zero_grad()

        props = model(src, tgt)
        loss = ml_criterion(props, tgt)

        loss.backward()
        ml_optimizer.step()
        if tf is not None:
            add_summary_value("pre-train loss", loss.data[0])
            tf_step += 1

            if tf_step % 100 == 0:
                tf_summary_writer.flush()


def reinforce():
    if tf:
        global tf_step
    for src, tgt in tqdm(training_data,
                         mininterval=1,
                         desc="Reinforce-train",
                         leave=False):
        rl_optimizer.zero_grad()

        max_words = model.sample(src)
        s_words, props = model.sample(src, False)

        reward = rouge_l(s_words, tgt)
        baseline = rouge_l(max_words, tgt)

        advantage = reward - baseline

        loss = rl_criterion(props, s_words, tgt, advantage)

        loss.backward()
        rl_optimizer.step()
        if tf is not None:
            add_summary_value("reinforce loss", loss.data[0])
            add_summary_value("reinforce advantage", advantage.mean().data)
            add_summary_value("reinforce baseline", baseline.mean().data)
            add_summary_value("reinforce reward", reward.mean().data)
            tf_step += 1

            if tf_step % 100 == 0:
                tf_summary_writer.flush()


try:
    if args.pretrain == "":
        print("=" * 40 + "Pre-train" + "=" * 40)
        model.train()
        if tf:
            tf_step = 0
        for step in range(args.pretrain_epochs):
            pre_train()
            model_state_dict = model.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict,
                "dict": data['dict']
            }
            torch.save(model_source, args.save.format(
                "Pre-train_" + str(step)))
    else:
        model_source = torch.load(args.pretrain)
        model.load_state_dict(model_source["model"])

    if tf:
        tf_step = 0
    print("=" * 40 + "Reinforce-train" + "=" * 40)
    for step in range(args.epochs):
        reinforce()
        model_state_dict = model.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict,
            "dict": data['dict']
        }
        torch.save(model_source, args.save.format(step))

except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early")
