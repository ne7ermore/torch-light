import argparse

parser = argparse.ArgumentParser(
    description='A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification')

parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--data', type=str, default='./data/corpus')

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--label_size', type=int, default=5)
parser.add_argument('--seed', type=int, default=614)
parser.add_argument('--lr', type=float, default=.0003)

parser.add_argument('--dropout', type=float, default=.5)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--rnn_hsz', type=int, default=256)
parser.add_argument('--beta1', type=float, default=.9)
parser.add_argument('--beta2', type=float, default=.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--lamda', type=float, default=.5)
parser.add_argument('--clip_norm', type=float, default=10.)

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


data = torch.load(args.data)
args.max_ori_len = data["max_ori_len"]
args.max_sum_len = data["max_sum_len"]
args.dict_size = data['dict']['dict_size']

print("=" * 30 + "arguments" + "=" * 30)
for k, v in args.__dict__.items():
    if k in ("epochs", "seed", "data"):
        pass
    print("{}: {}".format(k, v))
print("=" * 60)

from data_loader import DataLoader

training_data = DataLoader(
    data['train']['original'],
    data['train']['summary'],
    data['train']['label'],
    data['max_ori_len'],
    data['max_sum_len'],
    use_cuda,
    bsz=args.batch_size)

validation_data = DataLoader(
    data['test']['original'],
    data['test']['summary'],
    data['test']['label'],
    data['max_ori_len'],
    data['max_sum_len'],
    use_cuda,
    bsz=args.batch_size,
    evaluation=True,
    shuffle=False)

from model import *

model = Model(args, use_cuda)
if use_cuda:
    model = model.cuda()

optimizer = ScheduledOptim(
    torch.optim.Adam(model.parameters(), betas=(
        args.beta1, args.beta2), eps=args.eps),
    model.parameters(), args.lr, args.clip_norm)
nlp_critic = NlpCrossEntropy()
cls_critic = torch.nn.CrossEntropyLoss()


def evaluate():
    if tf:
        global tf_step
    model.eval()
    corrects = 0
    for original, _, label in validation_data:
        _, cls_props = model(original)
        corrects += (torch.max(cls_props, 1)
                     [1].view(label.size()).data == label.data).sum()

    return corrects, validation_data.sents_size


def train():
    if tf:
        global tf_step
    model.train()
    for original, summary, label in training_data:
        optimizer.zero_grad()

        summ_props, cls_props = model(original)
        cls_loss = cls_critic(cls_props, label)
        summ_loss = nlp_critic(summ_props, summary)
        loss = summ_loss + args.lamda * cls_loss
        loss.backward()

        optimizer.step()
        optimizer.clip_grad_norm()

        corrects = (torch.max(cls_props, 1)[1].view(
            label.size()).data == label.data).sum()

        if tf is not None:
            add_summary_value("cls_loss", cls_loss.data)
            add_summary_value("summ_loss", summ_loss.data)
            add_summary_value("loss", loss.data)
            add_summary_value("corrects", corrects)
            tf_step += 1

            if tf_step % 100 == 0:
                tf_summary_writer.flush()


try:
    for epoch in range(1, args.epochs + 1):
        train()
        optimizer.update_learning_rate()
        corrects, size = evaluate()
        print('-' * 90)
        print('| end of epoch {} | size {} | corrects {}'.format(
            epoch, size, corrects))
        print('-' * 90)

except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early")
