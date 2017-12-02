import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math

class highway_layer(nn.Module):
    def __init__(self, hsz, active):
        super().__init__()

        self.hsz = hsz
        self.active = active

        self.gate = nn.Linear(hsz, hsz)
        self.h = nn.Linear(hsz, hsz)

    def _init_weight(self):
        stdv = 1. / math.sqrt(self.hsz)

        self.gate.weight.data.uniform_(-stdv, stdv)
        self.gate.bias.data.fill_(-1)

        if active.__name__ == "relu":
            init.xavier_normal(self.h.weight)
        else:
            self.h.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        gate = F.sigmoid(self.gate(x))

        return torch.mul(self.active(self.h(x)), gate) + torch.mul(x, (1 - gate))

class Highway(nn.Module):
    def __init__(self, num_layers, hsz, active):
        super().__init__()

        self.layers = nn.ModuleList([
            highway_layer(hsz, active) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

if __name__ == "__main__":
    from torch.autograd import Variable

    x = Variable(torch.randn((2, 3)))
    hw = Highway(2, 3, F.relu)

    print(hw(x))

    x = Variable(torch.randn((2, 3)))
    hw = Highway(1, 3, F.tanh)

    print(hw(x))
