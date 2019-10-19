import numpy as np
import torch

import const


class DataLoader:
    def __init__(self, chars, words, sub_sidx, sub_eidx, obj_idxs, sub_slidx, sub_elidx, word2idx, char2idx, predicate2id, cuda=True, batch_size=64, shuffle=True):

        self.sents_size = len(chars)
        self.cuda = cuda
        self.bsz = batch_size
        self.step = 0
        self.stop_step = self.sents_size // batch_size

        self.word2idx = word2idx
        self.char2idx = char2idx
        self.predicate2id = predicate2id

        self.chars = np.asarray(chars)
        self.words = np.asarray(words)
        self.sub_sidx = np.asarray(sub_sidx)
        self.sub_eidx = np.asarray(sub_eidx)
        self.obj_idxs = np.asarray(obj_idxs)
        self.sub_slidx = np.asarray(sub_slidx)
        self.sub_elidx = np.asarray(sub_elidx)

        if shuffle:
            self.shuffle()

    def shuffle(self):
        index = np.arange(self.chars.shape[0])
        np.random.shuffle(index)

        self.chars = self.chars[index]
        self.words = self.words[index]
        self.sub_sidx = self.sub_sidx[index]
        self.sub_eidx = self.sub_eidx[index]
        self.obj_idxs = self.obj_idxs[index]
        self.sub_slidx = self.sub_slidx[index]
        self.sub_elidx = self.sub_elidx[index]

    def __iter__(self):
        return self

    def __next__(self):
        def data2tensor(insts, get_pos=False):
            max_len = max(len(inst) for inst in insts)

            inst_data = np.array(
                [inst + [const.PAD] * (max_len - len(inst)) for inst in insts])
            if get_pos:
                position = np.array(
                    [[pos_i+1 if w_i != const.PAD else 0 for pos_i, w_i in enumerate(inst)] for inst in inst_data])
                position = torch.from_numpy(position)

            inst_data = torch.from_numpy(inst_data)
            if self.cuda:
                inst_data = inst_data.cuda()

            if get_pos:
                position = position.cuda()
                return inst_data, position, max_len

            return inst_data

        def obj2tensor(objs, max_len):
            o1, o2 = np.zeros((len(objs), max_len, len(self.predicate2id))), np.zeros(
                (len(objs), max_len, len(self.predicate2id)))

            for idx, obj in enumerate(objs):
                for obj_s, obj_e, pred in obj:
                    o1[idx, obj_s, pred] = 1
                    o2[idx, obj_e, pred] = 1

            o1 = torch.from_numpy(o1)
            o2 = torch.from_numpy(o2)
            if self.cuda:
                o1 = o1.cuda()
                o2 = o2.cuda()

            return o1.float(), o2.float()

        def subl2tensor(subs):
            subs = torch.from_numpy(np.asarray(subs))
            if self.cuda:
                subs = subs.cuda()
            return subs.float()

        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        start = self.step * self.bsz
        self.step += 1

        chars, position, max_len = data2tensor(
            self.chars[start:start + self.bsz], True)
        words = data2tensor(self.words[start:start + self.bsz])
        sub_sidx = data2tensor(
            self.sub_sidx[start:start + self.bsz]).float().unsqueeze(2)
        sub_eidx = data2tensor(
            self.sub_eidx[start:start + self.bsz]).float().unsqueeze(2)
        obj_sidx, obj_eidx = obj2tensor(
            self.obj_idxs[start:start + self.bsz], max_len)
        sub_slidx = subl2tensor(self.sub_slidx[start:start + self.bsz])
        sub_elidx = subl2tensor(self.sub_elidx[start:start + self.bsz])

        return chars, words, position, sub_sidx, sub_eidx, obj_sidx, obj_eidx, sub_slidx, sub_elidx


if __name__ == "__main__":
    import os

    data = torch.load(os.path.join(const.DATAPATH, "corpus.new.pt"))
    dl = DataLoader(data["dev"]["char"],
                    data["dev"]["word"],
                    data["dev"]["sub_sidx"],
                    data["dev"]["sub_eidx"],
                    data["dev"]["obj_idxs"],
                    data["dev"]["sub_slidx"],
                    data["dev"]["sub_elidx"],
                    data["word2idx"],
                    data["char2idx"],
                    data["predicate2id"],
                    batch_size=2)

    chars, words, position, sub_sidx, sub_eidx, obj_sidx, obj_eidx, sub_slidx, sub_elidx = next(
        dl)

    print(chars.shape)
    print(words.shape)
    print(position.shape)
    print(sub_sidx.shape)
    print(sub_eidx.shape)
    print(obj_sidx.shape)
    print(obj_eidx.shape)
    print(sub_slidx.shape)
    print(sub_elidx.shape)
    '''
    torch.Size([2, 125])
    torch.Size([2, 125])
    torch.Size([2, 125])
    torch.Size([2, 125, 1])
    torch.Size([2, 125, 1])
    torch.Size([2, 125, 49])
    torch.Size([2, 125, 49])
    torch.Size([2, 1])
    torch.Size([2, 1])
    '''

    id2chars = {v: k for k, v in dl.char2idx.items()}
    id2words = {v: k for k, v in dl.word2idx.items()}

    print("".join([id2chars[idx] for idx in chars.tolist()[0]]))
    print(position[0])
    print(obj_sidx.ge(1).nonzero().tolist())
    print(obj_eidx.ge(1).nonzero().tolist())
    print(dl.predicate2id)
    print(sub_sidx.ge(1).nonzero().tolist())
