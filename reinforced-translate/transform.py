import torch
from torch.autograd import Variable

from model import Model
from corpus import normalizeString
from const import *


class Transform(object):
    def __init__(self, model_source="translate_Pre-train.pt"):
        model_source = torch.load(
            model_source, map_location=lambda storage, loc: storage)

        self.src_dict = model_source["dict"]["src"]
        self.idx2word = {v: k for k, v in model_source["dict"]["tgt"].items()}
        self.args = args = model_source["settings"]

        args.use_cuda = False
        model = Model(args)
        model.load_state_dict(model_source['model'])
        model = model.cpu()
        self.model = model.eval()

    def sent2tenosr(self, sentence):
        max_len = self.args.max_len - 2
        sentence = normalizeString(sentence)
        words = [w for w in sentence.strip().split()]

        if len(words) > max_len:
            words = words[:max_len]

        words = [WORD[BOS]] + words + [WORD[EOS]]
        idx = [self.src_dict[w] if w in self.src_dict else UNK for w in words]

        idx_data = torch.LongTensor(idx)
        idx_data_tensor = Variable(idx_data.unsqueeze(0), volatile=True)

        return idx_data_tensor

    def translate(self, sentence):
        idx_data = self.sent2tenosr(sentence)
        idx_data = idx_data.repeat(16, 1)
        words, _ = self.model(idx_data, False)

        return " ".join([self.idx2word[_id] for _id in words.data.tolist()[0]])


if __name__ == "__main__":
    t = Transform()
    en = t.translate("Tom est connu.")
    print(en)
