import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms

from PIL import Image

import numpy as np

from const import PAD

class Data_loader(object):
    def __init__(self, path, imgs, labels, max_len, batch_size, is_cuda, img_size=299, evaluation=False):
        self._path = path
        self._imgs = imgs
        self._labels = np.asarray(labels)
        self._max_len = max_len
        self._is_cuda = is_cuda
        self.evaluation = evaluation
        self._step = 0
        self._batch_size = batch_size
        self.sents_size = len(imgs)
        self._stop_step = self.sents_size // batch_size

        self._encode = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.CenterCrop(img_size),
                            transforms.ToTensor()
                        ])

    def __iter__(self):
        return self

    def __next__(self):
        def img2variable(img_files):
            tensors = [self._encode(Image.open(self._path + img_name).convert('RGB')).unsqueeze(0) for img_name in img_files]
            v = Variable(torch.cat(tensors, 0), volatile=self.evaluation)
            if self._is_cuda: v = v.cuda()
            return v

        def label2variable(labels):
            _labels = np.array([l + [PAD] * (self._max_len - len(l)) for l in labels])
            _labels = Variable(torch.from_numpy(_labels), volatile=self.evaluation)
            if self._is_cuda: _labels = _labels.cuda()
            return _labels

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        self._step += 1

        _imgs = img2variable(self._imgs[_start:_start+self._batch_size])
        _labels = label2variable(self._labels[_start:_start+self._batch_size])
        return _imgs, _labels

if __name__ == "__main__":
    data = torch.load("data/img_caption.pt")
    training_data = Data_loader(
                  "data/train2017/",
                  data['train']['imgs'],
                  data['train']['captions'],
                  16,batch_size=2,is_cuda=True)
    print(training_data.sents_size)
    img, labels = next(training_data)

    id2word = {v: k for k, v in data["dict"].items()}
    # print(img)
    print([id2word[_id] for _id in labels[1].data.tolist()])
