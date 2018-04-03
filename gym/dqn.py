import math
import random
from collections import namedtuple
from itertools import count

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
out_dim = env.action_space.n


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 32
GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
CAPACITY = 10000

torch.manual_seed(1234)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(1234)

if use_cuda:
    byteTensor = torch.cuda.ByteTensor
    tensor = torch.cuda.FloatTensor
    longTensor = torch.cuda.LongTensor
else:
    byteTensor = torch.ByteTensor
    tensor = torch.Tensor
    longTensor = torch.LongTensor


class DQN(nn.Module):
    def __init__(self, state_dim, out_dim, capacity, bsz, epsilon):
        super().__init__()
        self.steps_done = 0
        self.position = 0
        self.pool = []
        self.capacity = capacity
        self.bsz = bsz
        self.epsilon = epsilon

        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, out_dim)

        self.fc1.weight.data.uniform_(-.1, .1)
        self.fc2.weight.data.uniform_(-.1, .1)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        return self.fc2(x)

    def action(self, state):
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

        if random.random() > self.epsilon:
            return self(Variable(state, volatile=True)).data.max(1)[1].view(1, 1)
        else:
            return longTensor([[random.randrange(2)]])

    def push(self, *args):
        if len(self) < self.capacity:
            self.pool.append(None)
        self.pool[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.pool, self.bsz)

    def __len__(self):
        return len(self.pool)


dqn = DQN(state_dim, out_dim, CAPACITY, BATCH_SIZE, INITIAL_EPSILON)
if use_cuda:
    dqn = dqn.cuda()
optimizer = optim.Adam(dqn.parameters(), lr=0.0001)


def optimize_model():
    if len(dqn) < BATCH_SIZE:
        return
    transitions = dqn.sample()
    batch = Transition(*zip(*transitions))

    non_final_mask = byteTensor(
        tuple(map(lambda x: x is not None, batch.next_state)))
    non_final_next_states = Variable(
        torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(tensor))
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


perfect = 0

for _ in range(10000):
    state = env.reset()
    state = torch.from_numpy(state).type(tensor).view(1, -1)
    for t in count():
        action = dqn.action(state)
        next_state, reward, done, _ = env.step(action[0, 0])
        next_state = torch.from_numpy(
            next_state).type(tensor).view(1, -1)
        if done:
            next_state = None

        reward = tensor([reward])

        dqn.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        if done:
            if t > perfect:
                print(t)
                perfect = t
            break
