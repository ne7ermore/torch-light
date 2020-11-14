import pickle 
import os    

from torch.utils import data
import torch
import numpy as np

class Utterances(data.Dataset):
    def __init__(self, hparams):
        self.len_crop = hparams.len_crop
        self.train_dataset = pickle.load(open(os.path.join(hparams.data_dir, hparams.training_data), "rb"))
        self.num_tokens = len(self.train_dataset)
        
    def __getitem__(self, index):
        embedding, mel, f0 = self.train_dataset[index]
        
        if mel.shape[0] < self.len_crop:
            len_pad = self.len_crop - mel.shape[0]
            uttr = np.pad(mel, ((0,len_pad),(0,0)), 'constant')
            f0 = np.pad(f0, (0,len_pad), 'constant')
        elif tmp.shape[0] > self.len_crop:
            left = np.random.randint(mel.shape[0]-self.len_crop)
            uttr = mel[left:left+self.len_crop, :]
            f0 = f0[left:left+self.len_crop]
        else:
            uttr = mel
            f0 = f0
        
        return uttr, embedding, f0
    
    def __len__(self):
        return self.num_tokens

def get_loader(hparams, num_workers=0):    
    dataset = Utterances(hparams)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=hparams.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader