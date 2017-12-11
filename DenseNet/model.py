import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
        self.dropout = dropout

    def forward(self, input):
        out = self.layer(input)
        if self.dropout > 0.:
            out = F.dropout(out, p=self.dropout)
        return out

class _DenseBLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout):
        super().__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4*growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        self.dropout = dropout

    def forward(self, input):
        out = self.layer(input)
        out = torch.cat([out, input], 1)
        if self.dropout > 0.:
            out = F.dropout(out, p=self.dropout)
        return out

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, growth_rate, in_channels, dropout):
        super().__init__()

        self.bottleneck = nn.Sequential(OrderedDict([("dbl_{}".format(l),
            _DenseBLayer(in_channels + growth_rate*l, growth_rate, dropout)) for l in range(num_layers)]))

    def forward(self, input):
        return self.bottleneck(input)

class DenseNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.init_cnn_layer = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, args.channels, kernel_size=3, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(args.channels)),
            ('relu0', nn.ReLU(inplace=True))
        ]))

        denseblocks = []
        for l, nums in enumerate(args.layer_nums):
            denseblocks += [("db_{}".format(l), _DenseBlock(nums, args.growth_rate, args.channels, args.dropout))]
            _in_channels = args.channels + args.growth_rate*nums
            args.channels = _in_channels // 2
            if l != len(args.layer_nums)-1:
                denseblocks += [("t_{}".format(l), _Transition(_in_channels, args.channels, args.dropout))]

        denseblocks += [("nb_5", nn.BatchNorm2d(_in_channels))]
        denseblocks += [("relu_5", nn.ReLU(inplace=True))]

        if args.dropout != 0.:
            denseblocks += [("dropout_5", nn.Dropout(args.dropout))]

        self.denseblocks = nn.Sequential(OrderedDict(denseblocks))

        self.lr = nn.Linear(_in_channels, args.num_class)
        self.lr.bias.data.fill_(0)

    def forward(self, input):
        out = self.init_cnn_layer(input)
        out = self.denseblocks(out)
        out = F.avg_pool2d(out, 8).squeeze()
        return self.lr(out)
