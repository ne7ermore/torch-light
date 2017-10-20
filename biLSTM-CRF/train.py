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
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')

parser.add_argument('--save', type=str, default='model/lstm_crf_size.pt',
                    help='path to save the final model')
parser.add_argument('--save-epoch', action='store_true',
                    help='save every epoch')
parser.add_argument('--data', type=str, default='./data/lstm_crf_size.pt',
                    help='location of the data corpus')

parser.add_argument('--embed-dim', type=int, default=256,
                    help='number of embedding dimension [default: 256]')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='the probability for dropout (0 = no dropout) [default: 0.3]')
parser.add_argument('--lstm-hsz', type=int, default=256,
                    help='BiLSTM hidden size')
parser.add_argument('--lstm-layers', type=int, default=3,
                    help='biLSTM layer numbers')
parser.add_argument('--w-init', type=float, default=0.25,
                    help='weight init scope')
parser.add_argument('--n-warmup-steps', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# ##############################################################################
# Load data
# ##############################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_len"]
args.tag_size = data["tag_size"]
args.vocab_size = data['dict']['vocab_size']
args.trains_score = data['trains_score']
training_data = DataLoader(
             data['train']['src'],
             data['train']['label'],
             args.max_len,
             batch_size=args.batch_size,
             cuda=use_cuda)

validation_data = DataLoader(
              data['valid']['src'],
              data['valid']['label'],
              args.max_len,
              batch_size=args.batch_size,
              shuffle=False,
              cuda=use_cuda)

args.n_warmup_steps = args.n_warmup_steps if args.n_warmup_steps != 0 else training_data.sents_size // args.batch_size

# ##############################################################################
# Build model
# ##############################################################################
from model import BiLSTM_CRF_Size
from optim import ScheduledOptim

model = BiLSTM_CRF_Size(args)
if use_cuda:
   model = model.cuda()

optimizer = ScheduledOptim(
            torch.optim.Adam(model.parameters(),
                            betas=(0.9, 0.98), eps=1e-09),
            args.lstm_hsz, args.n_warmup_steps)

criterion = torch.nn.CrossEntropyLoss()

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm

def evaluate():
    model.eval()
    corrects = eval_loss = 0
    _size = validation_data.sents_size
    for batch in tqdm(range(0, _size, args.batch_size),
                      mininterval=0.2, desc='Evaluate Processing', leave=False):
        src, label = validation_data.get_batch(batch, evaluation=True)
        pred = model(src)
        loss = criterion(pred, label)
        eval_loss += loss.data[0]
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

    _size *= args.max_len
    return eval_loss/_size, corrects, corrects/_size * 100.0, _size

def train():
    model.train()
    total_loss = 0
    _size = training_data.sents_size
    for batch in tqdm(range(0, _size, args.batch_size),
                      mininterval=2, desc='Training Processing', leave=False):
        src, label = training_data.get_batch(batch)

        optimizer.zero_grad()
        pred = model(src)

        loss = criterion(pred, label)
        loss.backward()

        optimizer.step()
        optimizer.update_learning_rate()
        total_loss += loss.data
    return total_loss[0]/(_size*args.max_len)

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
        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))

        loss, corrects, acc, size = evaluate()
        epoch_start_time = time.time()
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc, corrects, size))
        print('-' * 90)

        model_state_dict = model.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict,
            "src_dict": data['dict']['src'],
            "trains_score": data['trains_score']
        }
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            torch.save(model_source, args.save)
        if args.save_epoch:
            torch.save(model_source, "{}_{}".format(args.save, epoch))

except KeyboardInterrupt:
    print("-"*80)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
