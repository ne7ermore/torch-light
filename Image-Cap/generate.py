import torch
from torch.autograd import Variable
from torchvision import transforms

from PIL import Image

import model

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
        self.encode = model.Encode(True)

        self._encode = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.RandomCrop(img_size),
                            transforms.ToTensor()
                        ])

    def Speak(self, img):
        enc = self.enc_img(img)
        hidden = self.actor.feed_enc(enc)
        _, words = self.actor(hidden)

        words = words.data.tolist()[0]

        return [self.idx2word[idx] for idx in words[1:]]

    def enc_img(self, imgs, path="data/val2017/"):
        tensors = [self._encode(Image.open(path + img_name).convert('RGB')).unsqueeze(0) for img_name in imgs]
        v = Variable(torch.cat(tensors, 0), volatile=True)
        v = v.cuda()

        return self.encode(v)[0]

if __name__ == "__main__":
    G = Gener("imgcapt_2.pt")

    print(G.Speak(["000000580294.jpg", "000000580294.jpg"]))