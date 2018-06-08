import argparse

parser = argparse.ArgumentParser(
    description='A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots')

parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--data', type=str, default='./data/corpus')

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=607)
parser.add_argument('--lr', type=float, default=.001)

parser.add_argument('--dropout', type=float, default=.5)
parser.add_argument('--emb_dim', type=int, default=200)
parser.add_argument('--first_rnn_hsz', type=int, default=200)
parser.add_argument('--fillters', type=int, default=8)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--match_vec_dim', type=int, default=50)
parser.add_argument('--second_rnn_hsz', type=int, default=50)

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


from data_loader import DataLoader

data = torch.load(args.data)
args.max_cont_len = data["max_cont_len"]
args.max_utte_len = data["max_utte_len"]
args.dict_size = data['dict']['dict_size']
args.kernel_size = (args.kernel_size, args.kernel_size)

print("=" * 30 + "arguments" + "=" * 30)
for k, v in args.__dict__.items():
    if k in ("epochs", "seed", "data"):
        continue
    print("{}: {}".format(k, v))
print("=" * 60)

training_data = DataLoader(
    data['train']['utterances'],
    data['train']['responses'],
    data['train']['labels'],
    data['max_cont_len'],
    data['max_utte_len'],
    use_cuda,
    bsz=args.batch_size)

validation_data = DataLoader(
    data['test']['utterances'],
    data['test']['responses'],
    data['test']['labels'],
    data['max_cont_len'],
    data['max_utte_len'],
    use_cuda,
    bsz=args.batch_size,
    shuffle=False,
    evaluation=True)

from model import Model

model = Model(args)
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    corrects = eval_loss = 0
    _size = validation_data.sents_size
    for utterances, responses, labels in validation_data:
        pred = model(utterances, responses)
        loss = criterion(pred, labels)

        eval_loss += loss.data[0]
        corrects += (torch.max(pred, 1)
                     [1].view(labels.size()).data == labels.data).sum()

    return eval_loss / _size, corrects, corrects / _size * 100.0, _size


def train():
    if tf:
        global tf_step
    model.train()
    for utterances, responses, labels in training_data:
        optimizer.zero_grad()

        pred = model(utterances, responses)
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()

        corrects = (torch.max(pred, 1)[1].view(
            labels.size()).data == labels.data).sum()

        if tf is not None:
            add_summary_value("loss", loss.data[0])
            add_summary_value("corrects", corrects)
            tf_step += 1

            if tf_step % 100 == 0:
                tf_summary_writer.flush()


try:
    print('-' * 90)
    for epoch in range(1, args.epochs + 1):
        train()
        print('-' * 90)
        loss, corrects, acc, size = evaluate()
        print('| end of epoch {:3d} | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(
            epoch, loss, acc, corrects, size))
        print('-' * 90)

except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early")
