import argparse

import torch

parser = argparse.ArgumentParser(description='DenseNet')
parser.add_argument('--save', type=str, default='./DenseNet.pt',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--unuse-cuda', action='store_true',
                    help='unuse cuda')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training')
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--dropout', type=float, default=0.)

parser.add_argument('--num-class', type=int, default=100)
parser.add_argument('--growth-rate', type=int, default=12)
parser.add_argument('--channels', type=int, default=24)
parser.add_argument('--depth', type=int, default=100)

args = parser.parse_args()

import torch

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and not args.unuse_cuda
args.layer_nums = [(100-4)//6 for _ in range(3)]
args.dropout = args.dropout if args.augmentation else 0.

if use_cuda:
    torch.cuda.manual_seed(args.seed)

# ##############################################################################
# Load data
###############################################################################
import torchvision.datasets as td
import torchvision.transforms as transforms
import numpy as np

def dataLoader(is_train=True, cuda=True, batch_size=64, shuffle=True):
        if is_train:
            trans = [transforms.RandomHorizontalFlip(),
                     transforms.RandomCrop(32, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]
            trans = transforms.Compose(trans)
            train_set = td.CIFAR100('data', train=True, transform=trans)
            size = len(train_set.train_labels)
            train_loader = torch.utils.data.DataLoader(
                            train_set, batch_size=batch_size, shuffle=shuffle)
        else:
            trans = [transforms.ToTensor(),
                     transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]
            trans = transforms.Compose(trans)
            test_set = td.CIFAR100('data', train=False, transform=trans)
            size = len(test_set.test_labels)
            train_loader = torch.utils.data.DataLoader(
                            test_set, batch_size=batch_size, shuffle=shuffle)

        return train_loader, size

training_data, training_size = dataLoader(cuda=use_cuda, batch_size=args.batch_size)
validation_data, validation_size = dataLoader(cuda=use_cuda, batch_size=args.batch_size, is_train=False, shuffle=False)
# ##############################################################################
# Build model
# ##############################################################################
import model
from optim import ScheduledOptim

from modelp import densenet161

densenet = model.DenseNet(args)

if use_cuda:
    densenet = densenet.cuda()

optimizer = ScheduledOptim(torch.optim.SGD(densenet.parameters(), lr=args.lr,
    momentum=args.momentum, weight_decay=args.weight_decay), args.epochs, args.lr)

criterion = torch.nn.CrossEntropyLoss()

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm

from torch.autograd import Variable

train_loss = []
valid_loss = []
train_acc = []
accuracy = []

def evaluate():
    densenet.eval()
    corrects = eval_loss = 0
    for data, label in tqdm(validation_data, mininterval=0.2,
                desc='Evaluate Processing', leave=False):
        data, label = Variable(data, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        pred = densenet(data)
        loss = criterion(pred, label)

        eval_loss += loss.data[0]
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

    return eval_loss/validation_size, corrects, corrects/validation_size * 100.0

def train():
    densenet.train()
    corrects = total_loss = 0
    for data, label in tqdm(training_data, mininterval=1,
                desc='Train Processing', leave=False):
        data, label = Variable(data), Variable(label)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()

        target = densenet(data)
        loss = criterion(target, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.data
        corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()

    return total_loss[0]/training_size, corrects, corrects/training_size * 100.0

# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        loss, corrects, acc = train()
        train_acc.append(acc / 100.)
        train_loss.append(loss*1000.)
        optimizer.update_learning_rate()
        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc, corrects, training_size))

        loss, corrects, acc = evaluate()
        valid_loss.append(loss*1000.)
        accuracy.append(acc / 100.)

        print('-' * 90)
        print('| end of epoch {:3d} | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, loss, acc, corrects, validation_size))
        print('-' * 90)
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            model_state_dict = densenet.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict
            }
            torch.save(model_source, args.save)
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

