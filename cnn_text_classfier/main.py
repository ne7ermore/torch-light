import os
import argparse
import time

import torch
import torch.autograd as autograd
import torch.nn.functional as F

useCuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=256,
                    help='number of epochs for train [default: 256]')
parser.add_argument('--batch-size', type=int, default=50,
                    help='batch size for training [default: 50]')
parser.add_argument('--log-interval',  type=int, default=1000,
                    help='report interval [default: 1000]')
parser.add_argument('--save', type=str, default='./cnn',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--max-len', type=int, default=10,
                    help='max length of one comment')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--embed-dim', type=int, default=256,
                    help='number of embedding dimension [default: 256]')
parser.add_argument('--kernel-num', type=int, default=250,
                    help='number of each kind of kernel')
parser.add_argument('--filter-sizes', type=str, default='1,2,3,4',
                    help='filter sizes')
parser.add_argument('--hidden-size', type=str, default='256,128,64',
                    help='hidden size')
parser.add_argument('--dropout-switches', type=str, default='t,f,t',
                    help='dropout-switches')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()
torch.manual_seed(args.seed)
if useCuda:
    torch.cuda.manual_seed(args.seed)

import model
import corpus

# Load data
###############################################################################
corpus = corpus.Corpus(path=args.data, max_len=args.max_len)
train_data, label_ids = corpus.get_data()
assert train_data.size(0) == label_ids.size(0)

# Build the model
# ##############################################################################
args.tokens = len(corpus.dictionary)
args.class_num = len(corpus.label)
args.filter_sizes = [int(fs) for fs in args.filter_sizes.split(",")]
args.hidden_size = [int(hs) for hs in args.hidden_size.split(",")]
args.dropout_switches = args.dropout_switches.split(",")
cnn = model.CNN_Text_C(args)
if useCuda:
    cnn = cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# Training
# ##############################################################################
def get_batch(source_data, source_ids, i, evaluation=False):
    assert len(source_data) == len(source_ids)
    batch_size = min(args.batch_size, len(source_data) - 1 - i)
    source_data = autograd.Variable(source_data[i:i+batch_size], volatile=evaluation)
    source_ids = autograd.Variable(source_ids[i:i+batch_size])
    if useCuda:
        return source_data.cuda(), source_ids.cuda()
    return source_data, source_ids

def evaluate(test_data, test_ids):
    cnn.eval()
    corrects, avg_loss = 0, 0
    data, label = autograd.Variable(test_data).cuda(), autograd.Variable(test_ids).cuda()
    target = cnn(data)
    loss = criterion(target, label)
    avg_loss += loss.data[0]
    corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()
    size = test_data.size(0)
    return avg_loss, corrects, corrects/size * 100.0, size

def train(train_data, label_ids, epoch):
    cnn.train()
    start_time, total_loss = time.time(), 0
    for batch, i in enumerate(range(0, label_ids.size(0)-1, args.batch_size)):
        data, label = get_batch(train_data, label_ids, i)
        optimizer.zero_grad()
        target = cnn(data)
        loss = criterion(target, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:d} batches | {:5.2f} ms/batch | loss {:5.4f}'.format(epoch, batch, train_data.size(0) // args.batch_size, elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

# Save Model
# ##############################################################################
best_val_loss = None
if not os.path.isdir(args.save): os.makedirs(args.save)

total_start_time = time.time()
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(train_data, label_ids, epoch)
        test_data, test_ids = corpus.get_testdata()
        avg_loss, corrects, accuracy, size = evaluate(test_data, test_ids)
        print('-' * 80)
        # print('| end of epoch {:3d} | time: {:5.2f}s | loss {:5.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, avg_loss, accuracy, corrects, size))
        print('| end of epoch {:3d} | time: {:5.2f}s | loss {:5.4f} | accuracy {:.4f}%'.format(epoch, time.time() - epoch_start_time, avg_loss, accuracy))
        print('-' * 80)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or avg_loss < best_val_loss:
            with open(os.path.join(args.save, "cnn_text_classfier.pt"), 'wb') as f:
                torch.save(cnn, f)
            best_val_loss = avg_loss
except KeyboardInterrupt:
    print("-"*80)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

corpus.save(args.save)
