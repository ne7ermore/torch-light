import time
from collections import deque
import random

import numpy as np
import torch

from net import *
from game import Game
from data_loader import DataLoader


class Train(object):
    def __init__(self,
                 use_cuda=USECUDA,
                 game_times=GAMETIMES,
                 eval_nums=EVALNUMS,
                 epochs=EPOCHS,
                 size=SIZE,
                 win_rate=WINRATE,
                 mini_batch=MINIBATCH,
                 lr=LR):

        if use_cuda:
            torch.cuda.manual_seed(1234)
        else:
            torch.manual_seed(1234)

        self.net = Net()
        self.eval_net = Net()
        if use_cuda:
            self.net = self.net.cuda()
            self.eval_net = self.eval_net.cuda()

        self.use_cuda = use_cuda
        self.win_rate = win_rate
        self.eval_nums = eval_nums
        self.game_times = game_times
        self.epochs = epochs
        self.size = size
        self.mini_batch = mini_batch
        self.dl = DataLoader(use_cuda, mini_batch)
        self.sample_data = deque(maxlen=TRAINLEN)

        self.gen_optim(lr)
        self.entropy = AlphaEntropy()

    def run(self):
        model_path = f"model_{time.strftime('%Y%m%d%H', time.localtime())}.pt"

        for step in range(1, 1 + self.epochs):
            draw_nums = [0]

            # init
            self.net.save_model(path=model_path)
            self.net.eval()

            self.eval_net.load_model(path=model_path, cuda=self.use_cuda)
            self.eval_net.eval()

            self.game = Game(self.net, self.eval_net)

            # train
            for i in range(self.game_times):
                self.sample(self.game.play(draw_nums))
                self.game.reset()
            self.dl(self.sample_data)
            self.train(step)
            self.optim.update_learning_rate(draw_nums[0], step)

            # eval
            result = [0, 0, 0]  # draw win loss
            for _ in range(self.eval_nums):
                self.game.evaluate(result)
                # self.game.board.show()
                self.game.reset()

            # save or reload model
            if result[1] + result[2] == 0:
                rate = 0
            else:
                rate = result[1] / (result[1] + result[2])

            print("evaluation")
            print(f"win - {result[1]}")
            print(f"loss - {result[2]}")
            print(f"draw - {result[0]}")

            if rate >= self.win_rate:
                print(f"new best model. rate - {rate}")
                self.net.save_model(path=model_path)
            else:
                print(
                    f"load last model. rate - {rate}")
                self.net.load_model(path=model_path, cuda=self.use_cuda)

            print("-" * 60 + "\r\n")

    def train(self, step):
        self.net.train()

        total_loss = 0.
        for states, pi, rewards in self.dl:
            self.optim.zero_grad()

            v, props = self.net(states)
            loss = self.entropy(props, v, pi, rewards)
            loss.backward()

            self.optim.step()

            total_loss += loss.item()

        print(f"training epoch - {step}")
        print(
            f"moves - [{len(self.sample_data)}]| loss - {total_loss/self.dl.stop_step}")
        print(f"learning rate - {self.optim.lr}")
        print("-" * 60 + "\r\n")

    def sample(self, datas):
        for state, pi, reward in datas:
            c_state = state.copy()
            c_pi = pi.copy()
            for i in range(4):
                c_state = np.array([np.rot90(s, i) for s in c_state])
                c_pi = np.rot90(c_pi.reshape(self.size, self.size), i)
                self.sample_data.append([c_state, c_pi.flatten(), reward])

                c_state = np.array([np.fliplr(s) for s in c_state])
                c_pi = np.fliplr(c_pi)
                self.sample_data.append([c_state, c_pi.flatten(), reward])

    def gen_optim(self, lr):
        optim = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=L2)
        self.optim = ScheduledOptim(optim, lr)


if __name__ == "__main__":
    t = Train()
    t.run()
