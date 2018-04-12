import argparse
import time

import torch

parser = argparse.ArgumentParser(description='Capsule classification')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--no_cuda', action='store_true')

parser.add_argument('--save', type=str, default='./capsule.pt')
parser.add_argument('--data', type=str, default='./data/corpus.pt')

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--hsz', type=int, default=64)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--num_primary_units', type=int, default=8)
parser.add_argument('--output_unit_size', type=int, default=16)
parser.add_argument('--primary_unit_size', type=int, default=1152)
parser.add_argument('--iterations', type=int, default=3)


args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and not args.no_cuda

# ##############################################################################
# Load data
################################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_len"]
args.vsz = data['dict']['vocab_size']
args.labels = data['dict']['label_size']
args.use_cuda = use_cuda

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
    cuda=use_cuda,
    evaluation=True)

# ##############################################################################
# Model
# ##############################################################################
import model

capsule = model.Capsule(args)
if use_cuda:
    capsule = capsule.cuda()

optimizer = torch.optim.Adam(capsule.parameters(), lr=args.lr)

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm


def evaluate():
    capsule.eval()
    corrects = eval_loss = 0
    _size = validation_data.sents_size
    for data, label in tqdm(validation_data, mininterval=0.2,
                            desc='Evaluate Processing', leave=False):
        props, lstm_feats = capsule(data)
        loss = capsule.loss(props, label, lstm_feats)

        eval_loss += loss.data[0]
        corrects += (torch.sqrt((props**2).sum(dim=2)).max(1)
                     [1].view(label.size()).data == label.data).sum()

    return eval_loss / _size, corrects, corrects / _size * 100.0, _size


def train():
    capsule.train()
    total_loss = 0
    for data, label in tqdm(training_data, mininterval=1,
                            desc='Train Processing', leave=False):
        optimizer.zero_grad()

        props, lstm_feats = capsule(data)
        loss = capsule.loss(props, label, lstm_feats)

        loss.backward()
        optimizer.step()

        total_loss += loss.data
    return total_loss[0] / training_data.sents_size * args.batch_size


# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        loss = train()

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(
            epoch, time.time() - epoch_start_time, loss))

        loss, corrects, acc, size = evaluate()

        epoch_start_time = time.time()
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(
            epoch, time.time() - epoch_start_time, loss, acc, corrects, size))
        print('-' * 90)
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            model_state_dict = capsule.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict,
                "src_dict": data['dict']['train']
            }
            torch.save(model_source, args.save)
except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early | cost time: {:5.2f}min".format(
        (time.time() - total_start_time) / 60.0))
