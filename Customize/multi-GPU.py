import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.parallel.data_parallel import DataParallel


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
            inst_data = np.array(inst + [0] * (max_len - len(inst)))
            inst_data_tensor = torch.from_numpy(inst_data)
            return inst_data_tensor

        data = pad_to_longest(self._src_sents[item], self._max_len)
        label = torch.tensor(self._label[item])

        return data, label


device = "cuda"

data = torch.load("corpus.pt")
ds = DS(data['train']['src'],
        data['train']['label'],
        1000)
train_data_loader = DataLoader(ds, batch_size=1000)

device_ids = [0, 7]

m = model(data['dict']['vocab_size'],
          data['dict']['label_size'],
          1000)
m = m.to(device)

optimizer = torch.optim.Adam(m.parameters())
criterion = torch.nn.CrossEntropyLoss()
m = DataParallel(m, device_ids=device_ids)


if __name__ == "__main__":
    m.train()
    for _ in range(100):
        for data, label in train_data_loader:
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            target = m(data)

            loss = criterion(target, label)
            loss.backward()
            optimizer.step()
