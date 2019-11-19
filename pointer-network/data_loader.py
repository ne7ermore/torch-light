import numpy as np
import torch

import const


class DataLoader(object):
    def __init__(self, src_texts, src_turn, tgt_indexs, tgt_texts, cuda, batch_size):
        self.cuda = cuda
        self.sents_size = len(src_texts)
        self._step = 0
        self.stop_step = self.sents_size // batch_size
        self._batch_size = batch_size
        self.src_texts = np.asarray(src_texts)
        self.src_turn = np.asarray(src_turn)
        self.tgt_indexs = np.asarray(tgt_indexs)
        self.tgt_texts = np.asarray(tgt_texts)

    def __iter__(self):
        return self

    def __next__(self):
        def pad_to_longest(insts):
            max_len = max(len(inst) for inst in insts)
            inst_data = np.array(
                [inst + [const.PAD] * (max_len - len(inst)) for inst in insts])
            inst_position = np.array(
                [[pos_i+1 if w_i != const.PAD else 0 for pos_i, w_i in enumerate(inst)] for inst in inst_data])

            inst_data_tensor = torch.from_numpy(inst_data)
            inst_position_tensor = torch.from_numpy(inst_position)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor.long(), inst_position_tensor.long(), max_len

        def index_pairs(t_indexs):
            max_len = max(len(inst) for inst in t_indexs)
            indexs = np.array([inst.tolist() + [const.PAD]
                               * (max_len - len(inst)) for inst in t_indexs])
            indexs = torch.from_numpy(indexs)
            if self.cuda:
                indexs = indexs.cuda()

            return indexs.long()

        def turns2tensor(turns, src_max_len):
            turns_data = np.array(
                [inst + [const.PAD] * (src_max_len - len(inst)) for inst in turns])
            turns_data = torch.from_numpy(turns_data)
            if self.cuda:
                turns_data = turns_data.cuda()
            return turns_data.long()

        if self._step == self.stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = self._batch_size
        self._step += 1

        src_tensor, src_postion, src_max_len = pad_to_longest(
            self.src_texts[_start:_start+_bsz])
        tgt_tensor, tgt_postion, tgt_max_len = pad_to_longest(
            self.tgt_texts[_start:_start+_bsz])

        tgt_indexs_tensor = index_pairs(self.tgt_indexs[_start:_start+_bsz])
        turns_tensor = turns2tensor(
            self.src_turn[_start:_start+_bsz], src_max_len)

        return (src_tensor, src_postion, turns_tensor), (tgt_tensor, tgt_postion), tgt_indexs_tensor
