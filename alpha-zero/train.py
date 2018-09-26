import time
from collections import deque
import random

import numpy as np
import torch

from net import *
from game import Game
from data_loader import DataLoader


class Train(object):
    def __init__(self, use_cuda=USECUDA, lr=LR):

        if use_cuda:
            torch.cuda.manual_seed(1234)
        else:
            torch.manual_seed(1234)

        self.kl_targ = 0.02
        self.lr_multiplier = 1.
        self.use_cuda = use_cuda

        self.net = Net()
        self.eval_net = Net()
        if use_cuda:
            self.net = self.net.cuda()
            self.eval_net = self.eval_net.cuda()

        self.dl = DataLoader(use_cuda, MINIBATCH)
        self.sample_data = deque(maxlen=TRAINLEN)
        self.gen_optim(lr)
        self.entropy = AlphaEntropy()

    def sample(self, datas):
        for state, pi, reward in datas:
            c_state = state.copy()
            c_pi = pi.copy()
            for i in range(4):
                c_state = np.array([np.rot90(s, i) for s in c_state])
                c_pi = np.rot90(c_pi.reshape(SIZE, SIZE), i)
                self.sample_data.append([c_state, c_pi.flatten(), reward])

                c_state = np.array([np.fliplr(s) for s in c_state])
                c_pi = np.fliplr(c_pi)
                self.sample_data.append([c_state, c_pi.flatten(), reward])

        return len(datas)

    def gen_optim(self, lr):
        optim = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=L2)
        self.optim = ScheduledOptim(optim, lr)

    def run(self):
        model_path = f"model_{time.strftime('%Y%m%d%H%M', time.localtime())}.pt"
        self.net.save_model(path=model_path)
        self.eval_net.load_model(path=model_path, cuda=self.use_cuda)

        for step in range(1, 1 + GAMETIMES):
            game = Game(self.net, self.eval_net)
            print(f"Game - {step} | data length - {self.sample(game.play())}")
            if len(self.sample_data) < MINIBATCH:
                continue

            states, pi, rewards = self.dl(self.sample_data)
            _, old_props = self.net(states)

            for _ in range(EPOCHS):
                self.optim.zero_grad()

                v, props = self.net(states)
                loss = self.entropy(props, v, pi, rewards)
                loss.backward()

                self.optim.step()

                _, new_props = self.net(states)
                kl = torch.mean(torch.sum(
                    torch.exp(old_props) * (old_props - new_props), 1)).item()
                if kl > self.kl_targ * 4:
                    break

            if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
            elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

            self.optim.update_learning_rate(self.lr_multiplier)

            print(
                f"kl - {kl} | lr_multiplier - {self.lr_multiplier} | loss - {loss}")
            print("-" * 100 + "\r\n")

            if step % CHECKOUT == 0:
                result = [0, 0, 0]  # draw win loss
                for _ in range(EVALNUMS):
                    game.reset()
                    game.evaluate(result)

                if result[1] + result[2] == 0:
                    rate = 0
                else:
                    rate = result[1] / (result[1] + result[2])

                print(f"step - {step} evaluation")
                print(
                    f"win - {result[1]} | loss - {result[2]} | draw - {result[0]}")

                # save or reload model
                if rate >= WINRATE:
                    print(f"new best model. rate - {rate}")
                    self.net.save_model(path=model_path)
                    self.eval_net.load_model(
                        path=model_path, cuda=self.use_cuda)
                else:
                    print(f"load last model. rate - {rate}")
                    self.net.load_model(path=model_path, cuda=self.use_cuda)

                print("-" * 100 + "\r\n")


if __name__ == "__main__":
    t = Train()
    t.run()
