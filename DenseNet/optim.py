import numpy as np

class ScheduledOptim(object):
    def __init__(self, optimizer, epochs, lr):
        self.optimizer = optimizer
        self.n_current_epochs = 0
        self.lr = lr
        self.epochs = epochs

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        self.n_current_epochs += 1

        # Is divided by 10 at 50% and 75% of the total number of training epochs
        if self.n_current_epochs == (self.epochs//3) or self.n_current_epochs == (self.epochs//3)*2:
            self.lr = self.lr / 10
            print("| learning rate updated - {}".format(self.lr))
            print('-' * 90)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
