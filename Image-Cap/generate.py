import json

import torch
from torch.autograd import Variable
from torchvision import transforms

from PIL import Image

import model
from caption import normalizeString

class Gener(object):
    def __init__(self, model_source, img_size=299):
        model_source = torch.load(model_source)

        self.word2idx = model_source["dict"]
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        args = model_source["settings"]
        actor = model.Actor(args.vocab_size,
                    args.dec_hsz,
                    args.rnn_layers,
                    2,
                    args.max_len,
                    args.dropout,
                    True)

        actor.load_state_dict(model_source["model"])
        actor = actor.cuda()

        self.actor = actor.eval()

        self._encode = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.CenterCrop(img_size),
                            transforms.ToTensor()
                        ])

        self.max_len = args.max_len

    def Speak(self, img):
        enc = self.enc_img(img)
        hidden = self.actor.feed_enc(enc)
        _, words, _ = self.actor.speak(hidden)

        words = words.data.tolist()[0]

        s = ""
        for idx in words[1:]:
            s += self.idx2word[idx]
            s += " "

            if idx == 3: break

        return s

    def enc_img(self, imgs, path="data/train2017/"):
        tensors = [self._encode(Image.open(path + img_name).convert('RGB')).unsqueeze(0) for img_name in imgs]
        v = Variable(torch.cat(tensors, 0), volatile=True)
        v = v.cuda()
        return self.actor.encode(v)

    def get_imgs(self):
        def _cut(s, max_len):
            words = [w for w in normalizeString(s).strip().split()]
            if len(words) > max_len:
                words = words[:max_len]
            return words

        caps = json.loads(next(open("data/captions_train2017.json")))
        images, annotations = caps["images"], caps["annotations"]
        img_dict = {img["id"]: img["file_name"] for img in images}

        imgs, labels = [], []
        for anno in annotations:
            imgs.append(img_dict[anno["image_id"]])
            labels.append(_cut(anno["caption"], self.max_len))

        return imgs, labels

if __name__ == "__main__":
    G = Gener("imgcapt_v2_2.pt")

    # print(G.Speak(["000000435299.jpg", "000000188689.jpg"]))
    # print(G.Speak(["000000188689.jpg", "000000188689.jpg"]))
    # print(G.Speak(["000000190236.jpg", "000000188689.jpg"]))
    # print(G.Speak(["000000532058.jpg", "000000188689.jpg"]))
    # print(G.Speak(["000000481404.jpg", "000000188689.jpg"]))
    # print(G.Speak(["000000256407.jpg", "000000188689.jpg"]))
    # print(G.Speak(["000000321557.jpg", "000000188689.jpg"]))

    # print(G.Speak(["000000322141.jpg", "000000322141.jpg"]))
    # print(G.Speak(["000000399932.jpg", "000000322141.jpg"]))
    imgs, labels = G.get_imgs()

    count = 0
    f = 200
    for img, l in zip(imgs[f:], labels[f:]):
        count += 1
        print(img)
        print(l)
        print(G.Speak([img, imgs[0]]))
        print("="*50)

        if count == 50: break