import math
import random
from collections import namedtuple
from itertools import count
from copy import deepcopy

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

env = gym.make('CartPole-v0').unwrapped
state_dim = env.observation_space.shape[0]
out_dim = env.action_space.n

env.reset()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 120
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
CAPACITY = 10000


class DQN(nn.Module):
    def __init__(self, state_dim, out_dim, capacity, bsz):
        super().__init__()
        self.steps_done = 0
        self.position = 0
        self.pool = []
        self.capacity = capacity
        self.bsz = bsz

        self.decode = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.decode(x)

    def action(self, state):
        eps_threshold = EPS_END + \
            (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            return self(Variable(state, volatile=True)).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(2)]])

    def push(self, *args):
        if len(self) < self.capacity:
            self.pool.append(None)
        self.pool[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.pool, self.bsz)

    def __len__(self):
        return len(self.pool)


dqn = DQN(state_dim, out_dim, CAPACITY, BATCH_SIZE)
optimizer = optim.RMSprop(dqn.parameters())


def optimize_model():
    if len(dqn) < BATCH_SIZE:
        return
    transitions = dqn.sample()
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.ByteTensor(
        tuple(map(lambda x: x is not None, batch.next_state)))
    non_final_next_states = Variable(
        torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.Tensor))
    next_state_values[non_final_mask] = dqn(non_final_next_states).max(1)[0]
    next_state_values.volatile = False
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_values = dqn(state_batch).gather(1, action_batch)

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for _ in range(10):
    state = env.reset()
    state = torch.from_numpy(state).type(torch.Tensor).view(1, -1)
    for t in count():
        action = dqn.action(state)
        next_state, reward, done, _ = env.step(action[0, 0])
        next_state = torch.from_numpy(
            next_state).type(torch.Tensor).view(1, -1)
        reward = torch.Tensor([reward])
        if done:
            next_state = None

        dqn.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        if done:
            print(t)
            break

print('Complete')
