import os

import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms

from PIL import Image

class Data_loader(object):
    def __init__(self, path, img_size, batch_size, is_cuda):
        self._img_files = os.listdir(path)
        self._path = path
        self._is_cuda = is_cuda
        self._step = 0
        self._batch_size = batch_size
        self.sents_size = len(self._img_files)
        self._stop_step = self.sents_size // batch_size

        self._encode = transforms.Compose([
                            transforms.Scale(img_size),
                            transforms.RandomCrop(img_size),
                            transforms.ToTensor()
                        ])

    def __iter__(self):
        return self

    def __next__(self):
        def img2variable(img_files):
            tensors = [self._encode(Image.open(self._path + img_name)).unsqueeze(0)
                    for img_name in img_files]
            v = Variable(torch.cat(tensors, 0))
            if self._is_cuda: v = v.cuda()
            return v

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        self._step += 1

        return img2variable(self._img_files[_start:_start+self._batch_size])

    def gen_image(self, g_out, epoch):
        torchvision.utils.save_image(g_out.data, 'images/epoch_{}.jpg'.format(epoch))

if __name__ == "__main__":
    dl = Data_loader('data/', 64, 64, True)
    print(dl.sents_size)
    for img, bsz in dl:
        continue