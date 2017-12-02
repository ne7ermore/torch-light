import numpy as np
import torch

class ScheduledOptim(object):
    def __init__(self, optimizer, d_model, n_warmup_steps, parameters, clip):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.clip = clip
        self.parameters = parameters

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm(self.parameters, self.clip)

    def update_learning_rate(self):
        self.n_current_steps += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([np.power(self.n_current_steps, -0.5), np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
