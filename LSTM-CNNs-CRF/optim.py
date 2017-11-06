import numpy as np

class ScheduledOptim(object):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.n_current_steps = 1
        self.lr = lr

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        self.n_current_steps += 1
        new_lr = self.lr / self.n_current_steps

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
