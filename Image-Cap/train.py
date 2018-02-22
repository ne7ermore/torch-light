import argparse

parser = argparse.ArgumentParser(description='Image Cation')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--unuse_cuda', action='store_true')
parser.add_argument('--path', type=str, default='data/')
parser.add_argument('--data', type=str, default='data/img_caption.pt')
parser.add_argument('--save', type=str, default='imgcapt_{}.pt')
parser.add_argument('--pre_lr', type=float, default=5e-5)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--new_lr', type=float, default=5e-6)
parser.add_argument('--actor_epochs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--iterations', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--dec_hsz', type=int, default=512)
parser.add_argument('--rnn_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=.5)

args = parser.parse_args()

import torch

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and not args.unuse_cuda

if use_cuda:
    torch.cuda.manual_seed(args.seed)

# ##############################################################################
# Load data
################################################################################
from data_loader import Data_loader

data = torch.load(args.data)
args.max_len = data["max_word_len"]
args.dict = data["dict"]
args.vocab_size = data["vocab_size"]

training_data = Data_loader(
              "data/train2017/",
              data['train']['imgs'],
              data['train']['captions'],
              args.max_len,
              batch_size=args.batch_size,
              is_cuda=use_cuda)

validation_data = Data_loader(
              "data/val2017/",
              data['valid']['imgs'],
              data['valid']['captions'],
              args.max_len,
              batch_size=args.batch_size,
              is_cuda=use_cuda,
              evaluation=True)

# ##############################################################################
# Build model
# ##############################################################################
import model
from const import PAD
from optim import Optim

encode = model.Encode(use_cuda)
actor = model.Actor(args.vocab_size,
                    args.dec_hsz,
                    args.rnn_layers,
                    args.batch_size,
                    args.max_len,
                    args.dropout,
                    use_cuda)

critic = model.Critic(args.vocab_size,
                      args.dec_hsz,
                      args.rnn_layers,
                      args.batch_size,
                      args.max_len,
                      args.dropout,
                      use_cuda)

optim_pre_A = Optim(actor.parameters(), args.pre_lr, True)
optim_pre_C = Optim(critic.parameters(), args.pre_lr, True)

optim_A = Optim(actor.parameters(), args.lr, False, args.new_lr)
optim_C = Optim(critic.parameters(), args.lr, False, args.new_lr)

criterion_A = torch.nn.CrossEntropyLoss(ignore_index=PAD)
criterion_C = torch.nn.MSELoss()

if use_cuda:
    actor = actor.cuda()
    critic = critic.cuda()

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm

from torch.autograd import Variable

from rouge import rouge_l

def pre_train_actor():
    total_loss = 0.
    # for imgs, labels in tqdm(training_data,
    #         mininterval=1, desc="Pre-train Actor", leave=False):
    for imgs, labels in training_data:
        optim_pre_A.zero_grad()

        enc = encode(imgs)[0]
        hidden = actor.feed_enc(enc)
        target, _ = actor(hidden)

        loss = criterion_A(target.view(-1, target.size(2)), labels.view(-1))

        loss.backward()
        optim_pre_A.step()
        total_loss += loss.data

    return total_loss[0]/training_data.sents_size

def pre_train_critic():
    iterations, total_loss = 0, .0
    actor.eval()
    critic.train()
    # for imgs, labels in tqdm(training_data,
    #         mininterval=1, desc="Pre-train Critic", leave=False):
    for imgs, labels in training_data:
        optim_pre_C.zero_grad()

        enc = encode(imgs)[0]

        hidden_A = actor.feed_enc(enc)
        props_A, words_A = actor(hidden_A)

        fixed_props_A = Variable(props_A.data.new(*props_A.size()), requires_grad=False)
        fixed_props_A.data.copy_(props_A.data)

        hidden_C = critic.feed_enc(enc)
        props_C, words_C = critic(words_A, hidden_C)

        scores_A, scores_C = rouge_l(words_A[:, 1:], labels), rouge_l(words_C, labels)

        loss = critic.td_error(scores_A, scores_C, fixed_props_A, props_C, criterion_C)
        loss.backward()

        optim_pre_C.step()
        total_loss += loss.data

        iterations += 1

        if iterations == args.iterations: break

    return total_loss[0]/args.iterations

def train_actor_critic():
    loss_A = loss_C = .0
    actor.train()
    critic.train()

    # for imgs, labels in tqdm(training_data,
    #         mininterval=1, desc="Actor-Critic Training", leave=False):
    for imgs, labels in training_data:
        optim_A.zero_grad()
        optim_C.zero_grad()

        enc = encode(imgs)[0]

        hidden_A = actor.feed_enc(enc)
        props_A, words_A = actor(hidden_A)

        fixed_props_A = Variable(props_A.data.new(*props_A.size()), requires_grad=False)
        fixed_props_A.data.copy_(props_A.data)

        hidden_C = critic.feed_enc(enc)
        props_C, words_C = critic(words_A, hidden_C)

        scores_A, scores_C = rouge_l(words_A[:, 1:], labels), rouge_l(words_C, labels)

        loss_c = critic.td_error(scores_A, scores_C, fixed_props_A, props_C, criterion_C)
        loss_c.backward()
        optim_C.step()
        loss_C += loss_c.data

        base = (scores_A - scores_C).mean()
        loss_a = criterion_A(props_A.view(-1, props_A.size(2)), labels.view(-1))*base
        loss_a.backward()
        optim_A.step()
        loss_A += loss_a.data

    loss_A = loss_A[0]/training_data.sents_size
    loss_C = loss_C[0]/training_data.sents_size

    return loss_A, loss_C

def eval():
    actor.eval()
    eval_loss = eval_score = .0
    # for imgs, labels in tqdm(validation_data,
    #         mininterval=1, desc="Actor-Critic Training", leave=False):
    for imgs, labels in validation_data:
        enc = encode(imgs)[0]

        hidden = actor.feed_enc(enc)
        props, words = actor(hidden)

        loss = criterion_A(props.view(-1, props.size(2)), labels.view(-1))
        scores = rouge_l(words[:, 1:], labels)
        scores = scores.sum()

        eval_loss += loss.data
        eval_score += scores

    eval_loss = eval_loss[0]/validation_data.sents_size
    eval_score = eval_score[0]/validation_data.sents_size

    return eval_loss, eval_score

try:
    s_time = time.time()
    print("="*40 + "Pre-train Actor" + "="*40)
    actor.train()
    for step in range(args.actor_epochs):
        loss = pre_train_actor()
        print("-"*20 + "epoch-{} | loss: {:.4f} | time: {:2.2f}".format(step, loss, time.time()-s_time) + "-"*20)
        s_time = time.time()

        model_state_dict = actor.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict,
            "dict": data['dict']
        }
        torch.save(model_source, args.save.format("pret-actor_" + step))

    print("="*40 + "Pre-train Critic" + "="*40)
    loss = pre_train_critic()
    print("-"*20 + "pre-train critic | loss: {:.4f} | time: {:2.2f}".format(loss, time.time()-s_time) + "-"*20)
    s_time = time.time()

    print("="*40 + "Actor-Critic Training" + "="*40)
    for step in range(args.epochs):
        loss_A, loss_C = train_actor_critic()
        print("-"*20 + "epoch-{}-train | actor loss: {:.4f} | critic loss: {:.4f}| time: {:2.2f}".format(step, loss_A, loss_C, time.time()-s_time))
        s_time = time.time()

        eval_loss, eval_score = eval()
        print("-"*20 + "epoch-{}-eval | eval loss: {:.4f} | eval score: {:.4f}| time: {:2.2f}".format(step, eval_loss, eval_score, time.time()-s_time) + "-"*20)

        model_state_dict = actor.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict,
            "dict": data['dict']
        }
        torch.save(model_source, args.save.format(step))

except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early")

