import os
import random
import zipfile

import torch
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage.transform import resize

from utils import bbox_iou


class IMGProcess(object):
    def __init__(self, source,
                 use_cuda=True,
                 img_path="imgs",
                 batch_size=100,
                 img_size=416,
                 confidence=0.5,
                 rebuild=True,
                 result="result"):

        self.colors = source["pallete"]
        self.num_classes = source["num_classes"]
        self.classes = source["classes"]
        self.confidence = confidence
        self.rebuild = rebuild
        self.result = result
        self.use_cuda = use_cuda
        self.img_size = img_size
        self.font = ImageFont.truetype("arial.ttf", 15)
        self.imgs = [os.path.join(img_path, img)
                     for img in os.listdir(img_path)]
        self.sents_size = len(self.imgs)
        self.bsz = min(batch_size, len(self.imgs))
        self._step = 0
        self._stop_step = self.sents_size // self.bsz

    def _encode(self, x):
        encode = T.Compose([T.Resize((self.img_size, self.img_size)),
                            T.ToTensor()])

        return encode(x)

    def img2Var(self, imgs):
        self.imgs = imgs = [Image.open(img).convert('RGB') for img in imgs]
        imgs_dim = torch.FloatTensor([img.size for img in imgs]).repeat(1, 2)

        with torch.no_grad():
            tensors = [self._encode(img).unsqueeze(0) for img in imgs]
            vs = torch.cat(tensors, 0)
            if self.use_cuda:
                vs = vs.cuda()
                imgs_dim = imgs_dim.cuda()

        return vs, imgs_dim

    def predict(self, prediction, nms_conf=0.4):
        """
        prediction:
            0:3 - x, y, h, w
            4 - confidence
            5: - class score
        """

        conf_mask = (prediction[:, :, 4] >
                     self.confidence).float().unsqueeze(2)
        prediction = prediction * conf_mask

        box_corner = prediction.new(*prediction.size())
        box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
        box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
        box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
        box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
        prediction[:, :, :4] = box_corner[:, :, :4]

        outputs = []

        for index, image_pred in enumerate(prediction):
            max_score, max_index = torch.max(
                image_pred[:, 5:], 1, keepdim=True)
            image_pred = torch.cat(
                (image_pred[:, :5], max_score, max_index.float()), 1)  # [10647, 7]

            non_zero_ind = (torch.nonzero(image_pred[:, 4])).view(-1)

            if non_zero_ind.size(0) == 0:
                continue

            image_pred_ = image_pred[non_zero_ind, :]
            img_classes = torch.unique(image_pred_[:, -1])

            objects, img_preds = [], []
            name = self.this_img_names[index].split("/")[-1]

            for c in img_classes:
                image_pred_class = image_pred_[image_pred_[:, -1] == c]

                _, conf_sort_index = torch.sort(
                    image_pred_class[:, 4], descending=True)
                image_pred_class = image_pred_class[conf_sort_index]

                max_detections = []
                while image_pred_class.size(0):
                    max_detections.append(image_pred_class[0].unsqueeze(0))
                    if len(image_pred_class) == 1:
                        break
                    ious = bbox_iou(max_detections[-1], image_pred_class[1:])
                    image_pred_class = image_pred_class[1:][ious < nms_conf]
                img_preds.append(torch.cat(max_detections))
                objects += [self.classes[int(x.squeeze()[-1])]
                            for x in max_detections]

            outputs.append((name, objects))
            img_preds = torch.cat(img_preds, dim=0)

            if self.rebuild:
                self.tensor2img(img_preds, index, name)

        return outputs

    def tensor2img(self, tensor, index, name):
        imgs_dim = self.imgs_dim[index] / self.img_size
        img = self.imgs[index]
        draw = ImageDraw.Draw(img)

        tensor[:, :4] = tensor[:, :4].clamp_(0, self.img_size) * imgs_dim
        for t in tensor:
            s_x, s_y, e_x, e_y = list(map(int, t[:4]))
            label = self.classes[int(t[-1])]
            color = random.choice(self.colors)
            draw.rectangle([s_x, s_y, e_x, e_y], outline=color)
            draw.text([s_x, s_y], label, fill=color, font=self.font)

        del draw

        img.save(os.path.join(self.result, "res_{}".format(name)))

    def __iter__(self):
        return self

    def __next__(self):
        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _s = self._step * self.bsz
        self._step += 1

        self.this_img_names = self.imgs[_s:_s + self.bsz]

        vs, self.imgs_dim = self.img2Var(self.this_img_names)

        return vs


class Data_loader(object):
    def __init__(self, label, path,
                 img_size=416,
                 max_objects=50,
                 batch_size=16,
                 is_cuda=True):

        self.datas = self._parse_label(label)
        self.max_objects = max_objects
        self.path = path
        self.img_size = img_size
        self.encode = T.Compose(
            [T.Resize((img_size, img_size)), T.ToTensor()])
        self.bsz = batch_size
        self.stop_step = len(self.datas) // batch_size + 1
        self._step = 0
        self.is_cuda = is_cuda

    def _parse_label(self, label):
        datas = []
        for f in os.listdir(label):
            img_name = f.replace('.txt', '.jpg')
            obj = []
            for line in open(os.path.join(label, f)):
                points = line.strip().split()
                obj.append([float(p) for p in points])
            datas.append([img_name, obj])

        return datas

    def _parse(self, data):
        img, obj = data
        img = os.path.join(self.path, img)
        img = np.array(Image.open(img).convert('RGB'))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else (
            (0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        input_img = resize(
            input_img, (self.img_size, self.img_size, 3), mode='reflect')
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()

        obj = np.asarray(obj)

        x1 = w * (obj[:, 1] - obj[:, 3] / 2)
        y1 = h * (obj[:, 2] - obj[:, 4] / 2)
        x2 = w * (obj[:, 1] + obj[:, 3] / 2)
        y2 = h * (obj[:, 2] + obj[:, 4] / 2)

        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]

        obj[:, 1] = ((x1 + x2) / 2) / padded_w
        obj[:, 2] = ((y1 + y2) / 2) / padded_h
        obj[:, 3] *= w / padded_w
        obj[:, 4] *= h / padded_h

        filled_labels = np.zeros((self.max_objects, 5))
        filled_labels[range(len(obj))[:self.max_objects]
                      ] = obj[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels).float()

        return input_img, filled_labels

    def __iter__(self):
        return self

    def __next__(self):
        if self._step == self.stop_step:
            self._step = 0
            raise StopIteration()

        start = self._step * self.bsz
        self._step += 1

        bsz = min(self.bsz, len(self.datas) - start)

        imgs, labels = [], []
        for data in self.datas[start:start + bsz]:
            input_img, filled_labels = self._parse(data)
            imgs.append(input_img)
            labels.append(filled_labels)

        imgs = torch.stack(imgs, dim=0)
        labels = torch.stack(labels, dim=0)

        if self.is_cuda:
            imgs, labels = imgs.cuda(), labels.cuda()

        return imgs, labels


if __name__ == "__main__":
    d = Data_loader("data/labels/val2014/", "data/val2014")
    imgs, labels = next(d)
    print(imgs.shape)
    print(labels.shape)
