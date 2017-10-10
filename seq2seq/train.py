import argparse

import torch
import torch.autograd as autograd
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='seq2seq')
parser.add_argument('--epochs', type=int, default=32,
                    help='number of epochs for train [default: 32]')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training [default: 64]')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='the probability for dropout (0 = no dropout) [default: 0.1]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval',  type=int, default=1000,
                    help='report interval [default: 1000]')
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')
parser.add_argument('--not-by-word', action='store_true',
                    help='segment sentences not by word')

parser.add_argument('--save', type=str, default='model/attn_model',
                    help='path to save the final model')
parser.add_argument('--save-epoch', action='store_true',
                    help='save every epoch')
parser.add_argument('--data', type=str, default='./data/train',
                    help='location of the data corpus')
parser.add_argument('--eval-data', type=str, default='./data/test',
                    help='location of the eval data corpus')

parser.add_argument('--proj-share-weight', action='store_true',
                    help='share linear weight')
parser.add_argument('--embs-share-weight', action='store_true')
parser.add_argument('--d-model', type=int, default=512,
                    help='equal dimension of word embedding dim')
parser.add_argument('--d-inner-hid', type=int, default=2048)
parser.add_argument('--n-head', type=int, default=8)
parser.add_argument('--n-layers', type=int, default=6)
parser.add_argument('--n-warmup-steps', type=int, default=4000)

args = parser.parse_args()
torch.manual_seed(args.seed)
args.by_word = not args.not_by_word

use_cuda = torch.cuda.is_available() and args.cuda_able
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# ##############################################################################
# Load data
# ##############################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_token_seq_len = data["max_token_seq_len"]

training_data = DataLoader(
             data['train']['src'],
             data['train']['tgt'],
             batch_size=args.batch_size,
             cuda=use_cuda)

validation_data = DataLoader(
              data['valid']['src'],
              data['valid']['tgt'],
              batch_size=args.batch_size,
              shuffle=False,
              cuda=use_cuda)

args.src_vocab_size = data['dict']['src_size']
args.tgt_vocab_size = data['dict']['tgt_size']

# ##############################################################################
# Build model
# ##############################################################################
from modules import Transformer
from optim import ScheduledOptim

import const

model = Transformer(
        args.src_vocab_size,
        args.tgt_vocab_size,
        args.max_token_seq_len,
        proj_share_weight=args.proj_share_weight,
        embs_share_weight=args.embs_share_weight,
        d_model=args.d_model,
        emb_dim=args.d_model,
        d_inner_hid=args.d_inner_hid,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout)

optimizer = ScheduledOptim(
     torch.optim.Adam(model.get_trainable_parameters(),
               betas=(0.9, 0.98), eps=1e-09),
     args.d_model, args.n_warmup_steps)

def get_criterion(vocab_size):
   ''' With PAD token zero weight '''
   weight = torch.ones(vocab_size)
   weight[const.PAD] = 0
   return torch.nn.CrossEntropyLoss(weight, size_average=False)

crit = get_criterion(args.tgt_vocab_size)

if use_cuda:
   model = model.cuda()
   crit = crit.cuda()

# ##############################################################################
# Training
# ##############################################################################
import time

def get_performance(crit, pred, gold):
    ''' Apply label smoothing if needed '''
    gold = gold.contiguous().view(-1)
    loss = crit(pred, gold)
    pred = pred.max(1)[1]

    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(const.PAD).data).sum()

    return loss, n_correct

def evaluate():
    model.eval()
    total_loss = n_total_words = n_total_correct = 0
    for batch in range(0, validation_data._sents_size, args.batch_size):
        src, tgt = validation_data.get_batch(batch, evaluation=True)
        gold = tgt[0][:, 1:]

        pred = model(src, tgt)
        loss, n_correct = get_performance(crit, pred, gold)

        n_total_words += gold.data.ne(const.PAD).sum()
        n_total_correct += n_correct
        total_loss += loss.data

    return total_loss[0]/n_total_words, n_total_correct, n_total_words

def train(epoch):
    model.train()
    start_time = time.time()
    total_loss = n_total_words = n_total_correct = 0
    for batch, i in enumerate(range(0, training_data._sents_size, args.batch_size)):
        src, tgt = training_data.get_batch(i)
        gold = tgt[0][:, 1:]

        optimizer.zero_grad()
        pred = model(src, tgt)

        loss, n_correct = get_performance(crit, pred, gold)
        loss.backward()

        optimizer.step()
        optimizer.update_learning_rate()

        n_total_words += gold.data.ne(const.PAD).sum()
        n_total_correct += n_correct
        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / (args.log_interval*n_total_words)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:d} batches | {:5.2f} ms/batch | loss {:5.6f} | {}/{} correct/all words'.format(epoch, batch, training_data._sents_size // args.batch_size, elapsed * 1000 / args.log_interval, cur_loss, n_total_correct, n_total_words))
            start_time = time.time()
            total_loss = 0
            n_total_words = 0
            n_total_correct = 0

# ##############################################################################
# Save Model
# ##############################################################################
best_corrects = None
total_start_time = time.time()
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch)
        loss, corrects, n_words = evaluate()
        print('-' * 120)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | {}/{} correct/all words'.format(epoch, time.time() - epoch_start_time, loss, corrects, n_words))
        print('-' * 120)

        model_state_dict = model.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict,
            "src_dict": data['dict']['src'],
            "tgt_dict": data['dict']['tgt']
        }
        if not best_corrects or best_corrects < corrects:
            best_corrects = corrects
            torch.save(model_source, args.save)
        if args.save_epoch:
            torch.save(model_source, "{}_{}".format(args.save, epoch))

except KeyboardInterrupt:
    print("-"*80)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
