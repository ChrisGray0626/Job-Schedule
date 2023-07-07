# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/2
"""
import random

import gym
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CartPoleEnv(gym.Wrapper):
    def __init__(self):
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        super().__init__(env)
        self.count = 0

    def reset(self):
        state, _ = self.env.reset()
        self.count = 0
        return state

    def execute(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        is_over = terminated or truncated
        self.count += 1
        if self.count >= 200:
            is_over = True

        return state, reward, is_over, info

    def play(self, choose_action=None, play=False):
        if choose_action is None:
            choose_action = self.choose_random_action

        state = self.reset()
        is_over = False
        trajectory = []
        while not is_over:
            action = choose_action(state)
            next_state, reward, is_over, _ = self.execute(action)
            trajectory.append([state, reward, action, next_state, is_over])
            state = next_state
            # 打印动画
            if play and random.random() < 0.2:  # 跳帧
                self.show()

        return trajectory

    def choose_random_action(self, state):
        return self.env.action_space.sample()

    def show(self):
        plt.imshow(self.env.render())
        plt.show()


if __name__ == '__main__':
    pass
