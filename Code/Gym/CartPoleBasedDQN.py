# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/2
"""
import random
from collections import deque

import torch

from CartPoleSolution import CartPoleSolution


class CartPoleBasedDQN(CartPoleSolution):
    def __init__(self):
        super().__init__()
        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )
        self.next_value_model = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )
        self.next_value_model.load_state_dict(self.value_model.state_dict())

        self.trajectory = deque()

    def choose_action(self, state):
        if random.random() < 0.01:
            return random.choice([0, 1])

        state = torch.FloatTensor(state).reshape(1, 4)
        action = self.value_model.forward(state).argmax().item()

        return action

    def calc_value(self, state, action):
        value = self.value_model.forward(state)
        value = value.gather(dim=1, index=action)

        return value

    def calc_target(self, reward, next_state, is_over):
        with torch.no_grad():
            target = self.next_value_model.forward(next_state)
        # 取所有动作中分数最大的
        # [b, 11] -> [b, 1]
        target = target.max(dim=1)[0]
        target = target.reshape(-1, 1)

        target = target * 0.98
        target *= (1 - is_over)
        target += reward

        return target

    def play(self):
        count = 0
        MINIMUM_ADD_NUM = 200
        MAXIMUM_NUM = 10000
        while count < MINIMUM_ADD_NUM:
            trajectory = self.env.play(self.choose_action, False)
            self.trajectory += trajectory
            count += len(trajectory)
        while len(self.trajectory) > MAXIMUM_NUM:
            self.trajectory.popleft()

    def sample_trajectory(self):
        SAMPLE_NUM = 64
        trajectory = random.sample(self.trajectory, SAMPLE_NUM)
        return trajectory

    def train(self):
        optimizer = torch.optim.Adam(self.value_model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        # 玩N局游戏,每局游戏训练M次
        for epoch in range(500):
            self.play()
            # states -> [b, 4]
            # rewards -> [b, 1]
            # actions -> [b, 1]
            # next_states -> [b, 4]
            # is_overs -> [b, 1]
            for i in range(200):
                trajectory = self.sample_trajectory()
                states, rewards, actions, next_states, is_overs = self.parse_trajectory(trajectory)
                # 计算values和targets
                # [b, 4] -> [b, 1]
                values = self.calc_value(states, actions)
                # [b, 4] -> [b, 1]
                targets = self.calc_target(rewards, next_states, is_overs)

                # 更新参数
                loss = loss_fn(values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 把model的参数复制给next_model
                if (i + 1) % 10 == 0:
                    self.next_value_model.load_state_dict(self.value_model.state_dict())

            if (epoch + 1) % 50 == 0:
                test_result = sum([self.test(play=False) for _ in range(10)]) / 10
                print(epoch + 1, test_result)


if __name__ == '__main__':
    cart_pole = CartPoleBasedDQN()
    cart_pole.train()
    cart_pole.test(play=False)
    pass
