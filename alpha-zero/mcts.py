from copy import deepcopy

import numpy as np

from const import *
from data_loader import to_tensor, to_numpy


class TreeNode(object):
    def __init__(self,
                 action=None,
                 props=None,
                 parent=None):

        self.parent = parent
        self.action = action
        self.children = []

        self.N = 0  # visit count
        self.Q = .0  # mean action value
        self.W = .0  # total action value
        self.P = props  # prior probability

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        index = np.argmax(np.asarray([c.uct() for c in self.children]))
        return self.children[index]

    def uct(self):
        return self.Q + self.P * CPUCT * (np.sqrt(self.parent.N) / (1 + self.N))

    def expand_node(self, props):
        self.children = [TreeNode(action=action, props=p, parent=self)
                         for action, p in enumerate(props) if p > 0.]

    def backup(self, v):
        self.N += 1
        self.W += v
        self.Q = self.W / self.N


class MonteCarloTreeSearch(object):
    def __init__(self, net,
                 ms_num=MCTSSIMNUM):

        self.net = net
        self.ms_num = ms_num

    def search(self, borad, node, temperature=.001):
        self.borad = borad
        self.root = node

        for _ in range(self.ms_num):
            node = self.root
            borad = self.borad.clone()

            while not node.is_leaf():
                node = node.select_child()
                borad.move(node.action)
                borad.trigger()

            # be carefull - opponent state
            value, props = self.net(
                to_tensor(borad.gen_state(), unsqueeze=True))
            value = to_numpy(value, USECUDA)[0]
            props = np.exp(to_numpy(props, USECUDA))

            # add dirichlet noise for root node
            if node.parent is None:
                props = self.dirichlet_noise(props)

            # normalize
            props[borad.invalid_moves] = 0.
            total_p = np.sum(props)
            if total_p > 0:
                props /= total_p

            # winner, draw or continue
            if borad.is_draw():
                value = 0.
            else:
                done = borad.is_game_over(player=borad.last_player)
                if done:
                    value = -1.
                else:
                    node.expand_node(props)

            while node is not None:
                value = -value
                node.backup(value)
                node = node.parent

        action_times = np.zeros(borad.size**2)
        for child in self.root.children:
            action_times[child.action] = child.N

        action, pi = self.decision(action_times, temperature)
        for child in self.root.children:
            if child.action == action:
                return pi, child

    @staticmethod
    def dirichlet_noise(props, eps=DLEPS, alpha=DLALPHA):
        return (1 - eps) * props + eps * np.random.dirichlet(np.full(len(props), alpha))

    @staticmethod
    def decision(pi, temperature):
        pi = (1.0 / temperature) * np.log(pi + 1e-10)
        pi = np.exp(pi - np.max(pi))
        pi /= np.sum(pi)
        action = np.random.choice(len(pi), p=pi)
        return action, pi
