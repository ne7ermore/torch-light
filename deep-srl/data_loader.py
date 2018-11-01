import numpy as np
import torch

import const


class DataLoader(object):
    def __init__(self, sents, label, cuda=True, batch_size=64, shuffle=True):
        self.cuda = cuda
        self.sents_size = len(sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size

        self.bsz = batch_size
        self._sents = np.asarray(sents)
        self._label = np.asarray(label)

        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._sents.shape[0])
        np.random.shuffle(indices)
        self._sents = self._sents[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def to_tensor(insts):
            max_len = max(len(inst) for inst in insts)
            inst_data = np.array(
                [inst + [const.PAD] * (max_len - len(inst)) for inst in insts])
            inst_data_tensor = torch.from_numpy(inst_data)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        start = self._step * self.bsz
        self._step += 1

        word = to_tensor(self._sents[start:start + self.bsz])
        label = to_tensor(self._label[start:start + self.bsz])
        return word, label.view(-1)


if __name__ == "__main__":
    data = torch.load("data/corpus.pt")
    training_data = DataLoader(
        data['train']['word'],
        data['train']['label'],
        cuda=False,
        batch_size=2,
        shuffle=False)
    print(data["dict"]["label"])
    word, label = next(training_data)
    word = word.tolist()
    d = {v: k for k, v in data["dict"]["word"].items()}
    for w in word:
        print([d[x] for x in w])
    print(label.shape[0])

    print(training_data.sents_size)
    print(training_data._sents.shape)
