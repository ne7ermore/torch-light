import math

import torch
import torch.nn as nn
import numpy as np

from utils import bbox_iou


class BasicConv(nn.Module):
    def __init__(self, ind, outd, kr_size, stride, padding, lr=0.1, bias=False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(ind, outd, kr_size, stride, padding, bias=bias),
            nn.BatchNorm2d(outd),
            nn.LeakyReLU(lr)
        )

    def forward(self, x):
        return self.layers(x)


class BasicLayer(nn.Module):
    def __init__(self, conv_1, conv_2, times):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(times):
            self.layers.append(BasicConv(*conv_1))
            self.layers.append(BasicConv(*conv_2))

    def forward(self, x):
        residual = x
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index % 2 == 1:
                x += residual
                residual = x

        return x


class BasicPred(nn.Module):
    def __init__(self,
                 structs,
                 use_cuda,
                 anchors,
                 classes,
                 height=416,
                 route_index=0):
        super().__init__()

        self.ri = route_index
        self.classes = classes
        self.height = height
        self.anchors = anchors
        self.torch = torch.cuda if use_cuda else torch

        # for train
        self.mse_loss = nn.MSELoss()  # Coordinate loss
        self.bce_loss = nn.BCELoss()  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

        in_dim = structs[0]
        self.layers = nn.ModuleList()
        for s in structs[1:]:
            if len(s) == 4:
                out_dim, kr_size, stride, padding = s
                layer = BasicConv(in_dim, out_dim, kr_size, stride, padding)
            else:
                out_dim, kr_size, stride, padding, _ = s
                layer = nn.Conv2d(in_dim, out_dim, kr_size, stride, padding)

            in_dim = out_dim
            self.layers.append(layer)

    def forward(self, x, targets=None):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if self.ri != 0 and index == self.ri:
                output = x

        detections = self.predict_transform(x, targets)

        if self.ri != 0:
            return detections, output
        else:
            return detections

    def predict_transform(self, inp, targets):
        """
        Code originally from https://github.com/eriklindernoren/PyTorch-YOLOv3.
        """
        def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
            """
            target - [bsz, max_obj, 5]
            """
            nB = target.size(0)
            nA = num_anchors
            nC = num_classes
            nG = grid_size
            mask = torch.zeros(nB, nA, nG, nG)
            conf_mask = torch.ones(nB, nA, nG, nG)
            tx = torch.zeros(nB, nA, nG, nG)
            ty = torch.zeros(nB, nA, nG, nG)
            tw = torch.zeros(nB, nA, nG, nG)
            th = torch.zeros(nB, nA, nG, nG)
            tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
            tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

            nGT = 0
            nCorrect = 0
            for b in range(nB):
                for t in range(target.shape[1]):
                    if target[b, t].sum() == 0:
                        # pad
                        continue
                    nGT += 1
                    # Convert to position relative to box
                    gx = target[b, t, 1] * nG
                    gy = target[b, t, 2] * nG
                    gw = target[b, t, 3] * nG
                    gh = target[b, t, 4] * nG
                    # Get grid box indices
                    gi = int(gx)
                    gj = int(gy)
                    # Get shape of gt box
                    gt_box = torch.FloatTensor(
                        np.array([0, 0, gw, gh])).unsqueeze(0)
                    # Get shape of anchor box
                    anchor_shapes = torch.FloatTensor(np.concatenate(
                        (np.zeros((len(anchors), 2)), np.array(anchors)), 1))

                    # Calculate iou between gt and anchor shapes
                    # 1 on 3
                    anch_ious = bbox_iou(gt_box, anchor_shapes)
                    # Where the overlap is larger than threshold set mask to zero (ignore)
                    conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
                    # Find the best matching anchor box

                    best_n = np.argmax(anch_ious)
                    # Get ground truth box
                    gt_box = torch.FloatTensor(
                        np.array([gx, gy, gw, gh])).unsqueeze(0)
                    # Get the best prediction
                    pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
                    # Masks
                    mask[b, best_n, gj, gi] = 1
                    conf_mask[b, best_n, gj, gi] = 1
                    # Coordinates
                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    # Width and height
                    tw[b, best_n, gj, gi] = math.log(
                        gw / anchors[best_n][0] + 1e-16)
                    th[b, best_n, gj, gi] = math.log(
                        gh / anchors[best_n][1] + 1e-16)
                    # One-hot encoding of label
                    target_label = int(target[b, t, 0])
                    tcls[b, best_n, gj, gi, target_label] = 1
                    tconf[b, best_n, gj, gi] = 1

                    # Calculate iou between ground truth and best matching prediction
                    iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                    pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
                    score = pred_conf[b, best_n, gj, gi]
                    if iou > 0.5 and pred_label == target_label and score > 0.5:
                        nCorrect += 1

            return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls

        # inp.shape - [bsz, 3*(num_classes+5), 13|26|52, 13|26|52]
        bsz = inp.size(0)
        grid_size = inp.size(2)
        stride = self.height // grid_size
        bbox_attrs = 5 + self.classes
        num_anchors = len(self.anchors)

        prediction = inp.view(bsz, num_anchors, bbox_attrs, grid_size,
                              grid_size).permute(0, 1, 3, 4, 2).contiguous()

        anchors = self.torch.FloatTensor(
            [(a[0] / stride, a[1] / stride) for a in self.anchors])

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size]).type(self.torch.FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size]).type(self.torch.FloatTensor)

        anchor_w = anchors[:, 0].view((1, num_anchors, 1, 1))
        anchor_h = anchors[:, 1].view((1, num_anchors, 1, 1))

        pred_boxes = self.torch.FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if targets is not None:

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=anchors.cpu().data,
                num_anchors=num_anchors,
                num_classes=self.classes,
                grid_size=grid_size,
                ignore_thres=0.5,
                img_dim=self.height)

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = mask.type(self.torch.ByteTensor)
            conf_mask = conf_mask.type(self.torch.ByteTensor)

            # Handle target variables
            with torch.no_grad():
                tx = tx.type(self.torch.FloatTensor)
                ty = ty.type(self.torch.FloatTensor)
                tw = tw.type(self.torch.FloatTensor)
                th = th.type(self.torch.FloatTensor)
                tconf = tconf.type(self.torch.FloatTensor)
                tcls = tcls.type(self.torch.LongTensor)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            loss_cls = self.ce_loss(
                pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            output = torch.cat(
                (
                    pred_boxes.view(bsz, -1, 4) * stride,
                    pred_conf.view(bsz, -1, 1),
                    pred_cls.view(bsz, -1, self.classes),
                ),
                -1,
            )
            return output
