import argparse
from itertools import count

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=543, metavar='N')
parser.add_argument('--render', action='store_true')

args = parser.parse_args()


env = gym.make('CartPole-v0').unwrapped
env.seed(args.seed)
torch.manual_seed(args.seed)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def select_action(self, state, values, select_props):
        state = torch.from_numpy(state).float()
        props, value = self(Variable(state))
        dist = Categorical(props)
        action = dist.sample()
        log_props = dist.log_prob(action)
        values.append(value)
        select_props.append(log_props)

        return action.data[0]


model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-2)


def main():
    for i_episode in count(1):
        state = env.reset()
        if args.render:
            env.render()
        values, select_props, policy_rewards = [], [], []
        for t in range(10000):
            action = model.select_action(state, values, select_props)
            state, reward, done, _ = env.step(action)
            policy_rewards.append(reward)

            if done:
                break

        R, rewards = 0, []
        for r in policy_rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)

        rewards = np.asarray(rewards)
        rewards = (rewards - rewards.mean()) / \
            (rewards.std() + np.finfo(np.float32).eps)

        value_loss, policy_loss = [], []
        for value, prop, r in zip(values, select_props, rewards):
            value_loss.append(F.smooth_l1_loss(
                value, Variable(torch.Tensor([r]))))
            reward = r - value.data[0]
            policy_loss.append(-prop * reward)

        loss = torch.cat(value_loss).sum() + torch.cat(policy_loss).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_episode % 10 == 0:
            print('Episode {}\tLast length: {:5d}\t'.format(
                i_episode, t))


if __name__ == '__main__':
    main()
