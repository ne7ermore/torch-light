import torch
import torch.nn as nn
import torch.nn.functional as F

from const import *


class ResBlockNet(nn.Module):
  def __init__(self,
               ind=RES_BLOCK_FILLTERS,
               block_filters=RES_BLOCK_FILLTERS,
               kr_size=3,
               stride=1,
               padding=1,
               bias=False):

    super().__init__()

    self.layers = nn.Sequential(
        nn.Conv2d(ind, block_filters, kr_size,
                  stride=stride,
                  padding=padding,
                  bias=bias),
        nn.BatchNorm2d(block_filters),
        nn.ReLU(),
        nn.Conv2d(block_filters, block_filters, kr_size,
                  stride=stride,
                  padding=padding,
                  bias=bias),
        nn.BatchNorm2d(block_filters),
    )

  def forward(self, x):
    res = x
    out = self.layers(x) + x

    return F.relu(out)


class Feature(nn.Module):
  def __init__(self,
               ind=IND,
               outd=RES_BLOCK_FILLTERS):

    super().__init__()

    self.layer = nn.Sequential(
        nn.Conv2d(ind, outd,
                  stride=1,
                  kernel_size=3,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(outd),
        nn.ReLU(),
    )
    self.encodes = nn.ModuleList([ResBlockNet() for _ in range(BLOCKS)])

  def forward(self, x):
    x = self.layer(x)
    for enc in self.encodes:
      x = enc(x)
    return x


class Policy(nn.Module):
  def __init__(self,
               ind=RES_BLOCK_FILLTERS,
               outd=OUTD,
               kernels=2):

    super().__init__()

    self.out = outd * kernels

    self.conv = nn.Sequential(
        nn.Conv2d(ind, kernels, kernel_size=1),
        nn.BatchNorm2d(kernels),
        nn.ReLU(),
    )

    self.linear = nn.Linear(kernels * outd, outd)
    self.linear.weight.data.uniform_(-.1, .1)

  def forward(self, x):
    x = self.conv(x)
    x = x.view(-1, self.out)
    x = self.linear(x)

    return F.log_softmax(x, dim=-1)


class Value(nn.Module):
  def __init__(self,
               ind=RES_BLOCK_FILLTERS,
               outd=OUTD,
               hsz=256,
               kernels=1):
    super().__init__()

    self.outd = outd

    self.conv = nn.Sequential(
        nn.Conv2d(ind, kernels, kernel_size=1),
        nn.BatchNorm2d(kernels),
        nn.ReLU(),
    )

    self.linear = nn.Sequential(
        nn.Linear(outd, hsz),
        nn.ReLU(),
        nn.Linear(hsz, 1),
        nn.Tanh(),
    )

    self._reset_parameters()

  def forward(self, x):
    x = self.conv(x)
    x = x.view(-1, self.outd)

    return self.linear(x)

  def _reset_parameters(self):
    for layer in self.modules():
      if type(layer) == nn.Linear:
        layer.weight.data.uniform_(-.1, .1)


class Net(nn.Module):
  def __init__(self):
    super().__init__()

    self.feat = Feature()
    self.value = Value()
    self.policy = Policy()

  def forward(self, x):
    feats = self.feat(x)
    winners = self.value(feats)
    props = self.policy(feats)

    return winners, props

  def save_model(self, path="model.pt"):
    torch.save(self.state_dict(), path)

  def load_model(self, path="model.pt", cuda=True):
    if cuda:
      self.load_state_dict(torch.load(path))
      self.cuda()
    else:
      self.load_state_dict(torch.load(
          path, map_location=lambda storage, loc: storage))
      self.cpu()


class AlphaEntropy(nn.Module):
  def __init__(self):
    super().__init__()
    self.v_loss = nn.MSELoss()

  def forward(self, props, v, pi, reward):
    v_loss = self.v_loss(v, reward)
    p_loss = -torch.mean(torch.sum(props * pi, 1))

    return p_loss + v_loss


class ScheduledOptim(object):
  def __init__(self, optimizer, lr):

    self.lr = lr
    self.optimizer = optimizer

  def step(self):
    self.optimizer.step()

  def zero_grad(self):
    self.optimizer.zero_grad()

  def update_learning_rate(self, lr_multiplier):
    new_lr = self.lr * lr_multiplier
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = new_lr


if __name__ == "__main__":
  net = Net()
  net.load_model()
