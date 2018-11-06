import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

PAD = 0


class model(nn.Module):
    def __init__(self, vocab_size, label_size, max_len, embed_dim=128, dropout=0.5):
        super().__init__()

        self.lookup_table = nn.Embedding(vocab_size, embed_dim)
        self.logistic = nn.Linear(embed_dim * max_len, label_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lookup_table(x)
        x = self.logistic(x.view(x.size(0), -1))
        x = self.dropout(x)
        return F.log_softmax(x, dim=-1)


class DS(Dataset):
    def __init__(self, src_sents, label, max_len):
        self.sents_size = len(src_sents)
        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._label = label

    def __len__(self):
        return self.sents_size

    def __getitem__(self, item):
        def pad_to_longest(inst, max_len):
            inst_data = np.array(inst + [PAD] * (max_len - len(inst)))
            inst_data_tensor = torch.from_numpy(inst_data)
            return inst_data_tensor

        data = pad_to_longest(self._src_sents[item], self._max_len)
        label = torch.tensor(self._label[item])

        return data, label


data = torch.load("corpus.pt")
ds = DS(data['train']['src'],
        data['train']['label'],
        data["max_len"])
train_data_loader = DataLoader(ds, batch_size=64, num_workers=3)
use_cuda = torch.cuda.is_available()

device_ids = [0, 1, 2]

cnn = model(data['dict']['vocab_size'],
            data['dict']['label_size'],
            data["max_len"])
cnn = cnn.cuda(device_ids[0])

optimizer = torch.optim.Adam(cnn.parameters())
optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
criterion = torch.nn.CrossEntropyLoss()


gpu_num = torch.cuda.device_count()
cnn = torch.nn.DataParallel(cnn, device_ids=device_ids).cuda()


if __name__ == "__main__":
    cnn.train()
    for _ in range(10):
        for data, label in train_data_loader:
            data = data.cuda(device_ids[0])
            label = label.cuda(device_ids[0])
            optimizer.zero_grad()
            target = cnn(data)

            loss = criterion(target, label)

            loss.backward()
            optimizer.module.step()
