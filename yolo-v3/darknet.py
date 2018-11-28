import torch
import torch.nn as nn

from collections import defaultdict

from layer import *
from utils import load_classes

OUT_DIM = 3 * (len(load_classes()) + 5)

DETECT_DICT = {
    'first': [1024, (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0), (1024, 3, 1, 1), (OUT_DIM, 1, 1, 0, 0)],
    'second': [768, (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1), (OUT_DIM, 1, 1, 0, 0)],
    'third': [384, (128, 1, 1, 0), (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), (OUT_DIM, 1, 1, 0, 0)],
}

LOSS_NAMES = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]


class LayerOne(BasicLayer):
    def __init__(self):
        super().__init__((64, 32, 1, 1, 0),
                         (32, 64, 3, 1, 1), 1)


class LayerTwo(BasicLayer):
    def __init__(self):
        super().__init__((128, 64, 1, 1, 0),
                         (64, 128, 3, 1, 1), 2)


class LayerThree(BasicLayer):
    def __init__(self):
        super().__init__((256, 128, 1, 1, 0),
                         (128, 256, 3, 1, 1), 8)


class LayerFour(BasicLayer):
    def __init__(self):
        super().__init__((512, 256, 1, 1, 0),
                         (256, 512, 3, 1, 1), 8)


class LayerFive(BasicLayer):
    def __init__(self):
        super().__init__((1024, 512, 1, 1, 0),
                         (512, 1024, 3, 1, 1), 4)


class FirstPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 classes,
                 route_index=4,
                 anchors=[(116, 90), (156, 198), (373, 326)]):
        super().__init__(structs, use_cuda, anchors, classes, route_index=route_index)


class SecondPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 classes,
                 route_index=4,
                 anchors=[(30, 61), (62, 45), (59, 119)]):
        super().__init__(structs, use_cuda, anchors, classes, route_index=route_index)


class ThirdPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 classes,
                 height=416,
                 anchors=[(10, 13), (16, 30), (33, 23)]):
        super().__init__(structs, use_cuda, anchors, classes)


class DarkNet(nn.Module):
    def __init__(self, use_cuda, nClasses):
        super().__init__()

        self.conv_1 = BasicConv(256, 512, 3, 2, 1)

        self.seq_1 = nn.Sequential(
            BasicConv(3, 32, 3, 1, 1),
            BasicConv(32, 64, 3, 2, 1),
            LayerOne(),
            BasicConv(64, 128, 3, 2, 1),
            LayerTwo(),
            BasicConv(128, 256, 3, 2, 1),
            LayerThree(),
        )
        self.seq_2 = nn.Sequential(
            BasicConv(512, 1024, 3, 2, 1),
            LayerFive()
        )

        self.layer_4 = LayerFour()

        self.uns_1 = nn.Sequential(
            BasicConv(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.uns_2 = nn.Sequential(
            BasicConv(256, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

        self.pred_1 = FirstPred(DETECT_DICT["first"], use_cuda, nClasses)
        self.pred_2 = SecondPred(DETECT_DICT["second"], use_cuda, nClasses)
        self.pred_3 = ThirdPred(DETECT_DICT["third"], use_cuda, nClasses)

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if type(layer) == nn.Conv2d:
                layer.weight.data.normal_(0.0, 0.02)

            if type(layer) == nn.BatchNorm2d:
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(self, x, targets=None):
        gather_losses = defaultdict(float)

        x = self.seq_1(x)
        r_0 = x

        x = self.layer_4(self.conv_1(x))
        r_1 = x

        x = self.seq_2(x)

        if targets is not None:
            (sum_loss, *losses), x = self.pred_1(x, targets)
            for name, loss in zip(LOSS_NAMES, losses):
                gather_losses[name] += loss
        else:
            det_1, x = self.pred_1(x)

        x = self.uns_1(x)
        x = torch.cat((x, r_1), 1)

        if targets is not None:
            (this_loss, *losses), x = self.pred_2(x, targets)
            sum_loss += this_loss
            for name, loss in zip(LOSS_NAMES, losses):
                gather_losses[name] += loss
        else:
            det_2, x = self.pred_2(x)

        x = self.uns_2(x)
        x = torch.cat((x, r_0), 1)

        if targets is not None:
            this_loss, *losses = self.pred_3(x, targets)
            sum_loss += this_loss
            for name, loss in zip(LOSS_NAMES, losses):
                gather_losses[name] += loss
            gather_losses["recall"] /= 3
            gather_losses["precision"] /= 3

            return sum_loss, gather_losses
        else:
            det_3 = self.pred_3(x)
            return torch.cat((det_1, det_2, det_3), 1)


if __name__ == "__main__":
    model_source = torch.load("yolo.v3.coco.weights.pt.old")
    model = DarkNet(False)
    model.load_state_dict(model_source['model'])

    par1 = model.seq_2.state_dict()

    model.seq_2 = nn.Sequential(
        BasicConv(512, 1024, 3, 2, 1),
        LayerFive()
    )
    model.pred_1 = FirstPred(DETECT_DICT["first"], False)

    new_sp = dict(model.seq_2.state_dict())
    new_pp = dict(model.pred_1.state_dict())

    for name, params in par1.items():
        if name[0] == "2":
            new_pp[name[2:]].data.copy_(params.data)
        else:
            new_sp[name].data.copy_(params.data)

    model.pred_1.load_state_dict(new_pp, False)
    model.seq_2.load_state_dict(new_sp, False)

    model_source['model'] = model.state_dict()
    torch.save(model_source, "yolo.v3.coco.weights.pt")
