from torch.optim import Adam
import torch

class Optim(object):
    def __init__(self, params, lr, is_pre, grad_clip, new_lr=0.0):
        self.optimizer = Adam(params, lr=lr, betas=(0.9, 0.98), eps=1e-09)
        self.grad_clip = grad_clip
        self.params = params
        if is_pre:
            self.step = self.pre_step
        else:
            assert new_lr != 0.0

            self.n_current_steps = 0
            self.new_lr = new_lr
            self.step = self.train_step

    def train_step(self):
        self.optimizer.step()

        self.n_current_steps += 1
        if self.n_current_steps == 1e6:
            self.update_learning_rate()

    def pre_step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm(self.params, self.grad_clip)

    def update_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.new_lr