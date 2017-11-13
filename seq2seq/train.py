import argparse

parser = argparse.ArgumentParser(description='seq2seq')
parser.add_argument('--epochs', type=int, default=32,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')

parser.add_argument('--save', type=str, default='seq2seq.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='data/seq2seq.pt',
                    help='location of the data corpus')

parser.add_argument('--not-share-linear', action='store_true',
                    help='Share the weight matrix between tgt word embedding/linear')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout)')
parser.add_argument('--d-model', type=int, default=512,
                    help='equal dimension of word embedding dim')
parser.add_argument('--d-ff', type=int, default=2048,
                    help='Position-wise Feed-Forward Networks inner layer dim')
parser.add_argument('--n-head', type=int, default=8)
parser.add_argument('--n-stack-layers', type=int, default=6)
parser.add_argument('--n-warmup-steps', type=int, default=0)

args = parser.parse_args()
args.share_linear = not args.not_share_linear

import torch
from torch.autograd import Variable

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and args.cuda_able
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# ##############################################################################
# Load data
# ##############################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_word_len = data["max_word_len"]

training_data = DataLoader(
             data['train']['src'],
             data['train']['tgt'],
             batch_size=args.batch_size,
             shuffle=False,
             cuda=use_cuda)

validation_data = DataLoader(
              data['valid']['src'],
              data['valid']['tgt'],
              batch_size=args.batch_size,
              shuffle=False,
              evaluation=True,
              cuda=use_cuda)

args.enc_vocab_size = data['dict']['src_size']
args.dec_vocab_size = data['dict']['tgt_size']

args.n_warmup_steps = args.n_warmup_steps if args.n_warmup_steps != 0 else training_data._stop_step

# ##############################################################################
# Build model
# ##############################################################################
from model import Transformer
from optim import ScheduledOptim
import const

model = Transformer(args)

optimizer = ScheduledOptim(
     torch.optim.Adam(model.get_trainable_parameters(),
               betas=(0.9, 0.98), eps=1e-09),
     args.d_model, args.n_warmup_steps)

def get_criterion(vocab_size):
   weight = torch.ones(vocab_size)
   weight[const.PAD] = 0
   return torch.nn.CrossEntropyLoss(weight, size_average=False)

crit = get_criterion(args.dec_vocab_size)

if use_cuda:
   model = model.cuda()
   crit = crit.cuda()

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm
import const

train_loss = []
valid_loss = []
accuracy = []

def get_performance(crit, pred, gold):
    gold = gold.contiguous().view(-1)

    loss = crit(pred, gold)
    pred = pred.max(1)[1]

    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(const.PAD).data).sum()

    return loss, n_correct

def evaluate():
    model.eval()
    total_loss = n_total_words = n_total_correct = 0
    for src, tgt in tqdm(validation_data, mininterval=0.2,
                desc='Evaluate Processing', leave=False):
        gold = tgt[0][:, 1:]

        pred = model(*src, *tgt)
        loss, n_correct = get_performance(crit, pred, gold)

        n_total_words += gold.data.ne(const.PAD).sum()
        n_total_correct += n_correct
        total_loss += loss.data

    return total_loss[0]/n_total_words, n_total_correct, n_total_words, n_total_correct/n_total_words

def train():
    model.train()
    start_time = time.time()
    total_loss = n_total_words = 0
    for src, tgt in tqdm(training_data, mininterval=1,
                desc='Train Processing', leave=False):

        gold = tgt[0][:, 1:]

        optimizer.zero_grad()
        pred = model(*src, *tgt)

        loss, _ = get_performance(crit, pred, gold)
        loss.backward()

        optimizer.step()
        optimizer.update_learning_rate()

        n_total_words += gold.data.ne(const.PAD).sum()
        total_loss += loss.data

    return total_loss[0]/n_total_words

# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()
try:
    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss)
        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))

        loss, corrects, n_words, acc = evaluate()
        valid_loss.append(loss)
        accuracy.append(acc)
        epoch_start_time = time.time()
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc*100, corrects, n_words))
        print('-' * 90)

        if not best_acc or best_acc < acc:
            best_acc = acc
            model_state_dict = model.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict,
                "src_dict": data['dict']['src'],
                "tgt_dict": data['dict']['tgt']
            }
            torch.save(model_source, args.save)
except KeyboardInterrupt:
    print("-"*80)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

print(train_loss)
print(valid_loss)
print(accuracy)
