import argparse

import torch
import torch.autograd as autograd
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='CNN Ranking')
parser.add_argument('--epochs', type=int, default=32,
                    help='number of epochs for train [default: 32]')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training [default: 64]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval',  type=int, default=1000,
                    help='report interval [default: 1000]')
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')

parser.add_argument('--save', type=str, default='model/cnn_ranking_model',
                    help='path to save the final model')
parser.add_argument('--save-epoch', action='store_true',
                    help='save every epoch')
parser.add_argument('--data', type=str, default='./data/pair_cnn.pt',
                    help='location of the data corpus')

parser.add_argument('--embed-dim', type=int, default=64,
                    help='number of embedding dimension [default: 64]')
parser.add_argument('--filter-sizes', type=str, default='2,3',
                    help='filter sizes')
parser.add_argument('--num-filters', type=int, default=64,
                    help='Number of filters per filter size [default: 64]')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--hidden-size', type=int, default=128,
                    help='hidden size')
parser.add_argument('--l_2', type=float, default=0.,
                    help="L2 regularizaion lambda [default: 0.0]")

args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able
if use_cuda:
    torch.cuda.manual_seed(args.seed)

args.filter_sizes = list(map(int, args.filter_sizes.split(",")))

# ##############################################################################
# Load data
# ##############################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_lenth_src = data["max_lenth_src"]
args.max_lenth_tgt = data["max_lenth_tgt"]

args.src_vocab_size = data['dict']['src_size']
args.tgt_vocab_size = data['dict']['tgt_size']

training_data = DataLoader(
             data['train']['src'],
             data['train']['tgt'],
             data['train']['label'],
             args.max_lenth_src,
             args.max_lenth_tgt,
             batch_size=args.batch_size,
             cuda=use_cuda)

validation_data = DataLoader(
              data['valid']['src'],
              data['valid']['tgt'],
              data['valid']['label'],
              args.max_lenth_src,
              args.max_lenth_tgt,
              batch_size=args.batch_size,
              shuffle=False,
              cuda=use_cuda)

# ##############################################################################
# Build model
# ##############################################################################
from module import CNN_Ranking

model = CNN_Ranking(args)
if use_cuda:
   model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# ##############################################################################
# Training
# ##############################################################################
import time

def evaluate():
    model.eval()
    corrects = eval_loss = 0
    _size = validation_data._sents_size
    for batch in range(0, _size, args.batch_size):
        src, tgt, label = validation_data.get_batch(batch, evaluation=True)
        pred = model(src, tgt)
        loss = criterion(pred, label)
        eval_loss += loss.data[0]
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

    return eval_loss/_size, corrects, corrects/_size * 100.0, _size

def train(epoch):
    model.train()
    start_time = time.time()
    total_loss = 0
    for batch, i in enumerate(range(0, training_data._sents_size, args.batch_size)):
        src, tgt, label = training_data.get_batch(i)

        optimizer.zero_grad()
        pred = model(src, tgt)

        loss = criterion(pred, label)
        loss.backward()

        optimizer.step()
        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:d} batches | {:5.2f} ms/batch | loss {:5.6f}'.format(epoch, batch, training_data._sents_size // args.batch_size, elapsed * 1000 / args.log_interval, cur_loss))
            start_time = time.time()
            total_loss = 0

# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch)
        loss, corrects, acc, size = evaluate()
        print('-' * 120)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc, corrects, size))
        print('-' * 120)

        model_state_dict = model.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict,
            "src_dict": data['dict']['src'],
            "tgt_dict": data['dict']['tgt']
        }
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            torch.save(model_source, args.save)
        if args.save_epoch:
            torch.save(model_source, "{}_{}".format(args.save, epoch))

except KeyboardInterrupt:
    print("-"*80)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
