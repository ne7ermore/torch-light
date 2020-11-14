import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from encode import Model, get_loader, set_logger, RAdam, Writer

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lambda_cd', type=float, default=1)
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16)
    parser.add_argument('--f0_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_iters', type=int, default=1000000)
    parser.add_argument('--len_crop', type=int, default=128)
    
    parser.add_argument('--log_step', type=int, default=1000)

    return parser.parse_args()    

class Train(object):
    def __init__(self, config):
        for k, v in config.__dict__.items():
            self.__setattr__(k, v)

        self.data_loader = get_loader(config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Model(config).to(self.device)
        self.optimizer = RAdam(self.model.parameters(), config.lr)

        self.writer = Writer(self.data_dir)
        self.logger = set_logger("Training", usefile=os.path.join(self.data_dir, "log"))
        self.logger.info(config)

    def save_checkpoint(self, p):
        torch.save({"model": self.model.state_dict()}, p)

    def training(self):
        self.model.train()

        for i in range(self.num_iters):
            try:
                x_real, emb_org, f0 = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org, f0 = next(data_iter)
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
            f0 = f0.to(self.device) 

            x_identic, x_identic_psnt, code_real = self.model(x_real, emb_org, emb_org, f0)
            g_loss_id = F.l1_loss(x_real, x_identic)   
            g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)   
            
            code_reconst = self.model(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)

            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()        

            for k, v in loss.items():
                self.writer.log_training(k, v, i)