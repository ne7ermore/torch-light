import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms as T

from PIL import Image

import numpy as np


class Img_loader(object):
    def __init__(self, path, bsz, is_cuda=True, img_size=224, evaluation=False):
        self.bsz = bsz
        self.data = [line.strip().split() for line in open(path)]
        self.ssz = len(self.data)
        self.stop_step = self.ssz // bsz
        self.step = 0
        self.is_cuda = is_cuda
        self.evaluation = evaluation

        self.encode = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])

    def __iter__(self):
        return self

    def __next__(self):
        def img2variable(datas):
            imgs, scores = zip(*datas)
            tensors = [self.encode(Image.open(
                "data/images/" + img).convert('RGB')).unsqueeze(0) for img in imgs]
            v = Variable(torch.cat(tensors, 0), volatile=self.evaluation)

            scores = np.asarray(list(map(float, scores)), dtype=np.float32)
            scores = Variable(torch.from_numpy(scores),
                              volatile=self.evaluation)

            if self.is_cuda:
                v = v.cuda()
                scores = scores.cuda()

            return v, scores

        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        start = self.step * self.bsz
        self.step += 1

        return img2variable(self.data[start:start + self.bsz])


if __name__ == "__main__":
    training_data = Img_loader("data/validation/test_1.txt", 2)
    i, s = next(training_data)
    print(i.shape)
    print(s.shape)
