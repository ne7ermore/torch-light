import argparse
import time

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='NLG for Chinese Poetry')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')

parser.add_argument('--save', type=str, default='./ch_poe.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/ch_pro_nlg.pt',
                    help='location of the data corpus')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout')
parser.add_argument('--embed-dim', type=int, default=128,
                    help='number of embedding dimension')
parser.add_argument('--hidden-size', type=int, default=128,
                    help='number of lstm hidden dimension')
parser.add_argument('--lstm-layers', type=int, default=3,
                    help='biLSTM layer numbers')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--bidirectional', action='store_true',
                    help='If True, becomes a bidirectional LSTM')

args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able

# ##############################################################################
# Load data
###############################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_word_len"]
args.vocab_size = data['dict']['src_size']

training_data = DataLoader(
                 data['train'],
                 args.max_len,
                 args.batch_size,
                 cuda=use_cuda)

# ##############################################################################
# Build model
# ##############################################################################
import model

rnn = model.Model(args)
if use_cuda:
    rnn = rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm

train_loss = []

def repackage_hidden(h):
    if type(h) == Variable:
        if use_cuda:
            return Variable(h.data).cuda()
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train():
    rnn.train()
    total_loss = 0
    hidden = rnn.init_hidden(args.batch_size)
    for data, label in tqdm(training_data, mininterval=1,
                desc='Train Processing', leave=False):
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        target, hidden = rnn(data, hidden)
        loss = criterion(target, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm(rnn.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data
    return total_loss[0]/training_data.sents_size

# ##############################################################################
# Save Model
# ##############################################################################
import generate
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        if use_cuda:
            rnn = rnn.cuda()
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))
        print('-' * 90)
        rnn.cpu()
        G = generate.Generate(model=rnn, src_dict=data['dict']['src'], args=args)
        print("generate - [{}]".format(G.Create(20)))
        print('-' * 90)

except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

model_state_dict = rnn.state_dict()
model_source = {
    "settings": args,
    "model": model_state_dict,
    "src_dict": data['dict']['src']
}
torch.save(model_source, args.save)
print(train_loss)

