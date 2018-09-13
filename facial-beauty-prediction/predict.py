import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np

from PIL import Image

from train import Beauty


class Predict:
    def __init__(self):
        model_source = torch.load(
            "./model.pt", map_location=lambda storage, loc: storage)
        model = Beauty()
        model.load_state_dict(model_source["model"])
        model.eval()
        self.model = model

        self.encode = T.Compose([
            T.Resize(224),
            T.ToTensor()
        ])

    def img2V(self, img):
        t = self.encode(Image.open(img).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            t = Variable(t)
        return t

    def divine(self, img):
        v = self.img2V(img)
        score = self.model(v)

        return round(score.data.tolist(), 3)


if __name__ == "__main__":
    p = Predict()
    score = p.divine("data/imgs/h2.jpeg")
    print(score)
