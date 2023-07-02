# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/2
"""
import numpy as np
import torch

from CartPoleEnv import CartPoleEnv


class CartPoleSolution:

    def __init__(self):
        self.env = CartPoleEnv()

    def choose_action(self, state):
        pass

    @staticmethod
    def parse_trajectory(trajectory):
        # [b, 4]
        states = np.array([i[0] for i in trajectory])
        states = torch.FloatTensor(states).reshape(-1, 4)
        # [b, 1]
        rewards = np.array([i[1] for i in trajectory])
        rewards = torch.FloatTensor(rewards).reshape(-1, 1)
        # [b, 1]
        actions = np.array([i[2] for i in trajectory])
        actions = torch.LongTensor(actions).reshape(-1, 1)
        # [b, 4]
        next_states = np.array([i[3] for i in trajectory])
        next_states = torch.FloatTensor(next_states).reshape(-1, 4)
        # [b, 1]
        is_overs = np.array([i[4] for i in trajectory])
        is_overs = torch.LongTensor(is_overs).reshape(-1, 1)

        return states, rewards, actions, next_states, is_overs

    def test(self, play):
        trajectory = self.env.play(self.choose_action, play)
        reward_sum = sum([i[1] for i in trajectory])

        return reward_sum

    def train(self):
        pass


if __name__ == '__main__':
    pass
