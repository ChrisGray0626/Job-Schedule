# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/2
"""
import random

import numpy as np

from CartPoleEnv import CartPoleEnv
import torch


class CartPoleBasedDQN:
    def __init__(self):
        self.env = CartPoleEnv()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Softmax(dim=1),
        )
        self.model_td = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

        self.model.forward(torch.randn(2, 4)), self.model_td.forward(torch.randn(2, 4))
        self.trajectory = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).reshape(1, 4)
        # [1, 4] -> [1, 2]
        prob = self.model.forward(state)
        # 根据概率选择一个动作
        action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]

        return action

    @staticmethod
    def sample_data(trajectory):
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

    @staticmethod
    def calc_advantage(deltas):
        advantages = []
        # 反向遍历 deltas
        s = 0.0
        for delta in deltas[::-1]:
            s = 0.98 * 0.95 * s + delta
            advantages.append(s)
        # 逆序
        advantages.reverse()

        return advantages

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer_td = torch.optim.Adam(self.model_td.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()
        # 玩N局游戏,每局游戏训练M次
        for epoch in range(500):
            # 玩一局游戏,得到数据
            trajectory = self.env.play(self.choose_action, False)
            # states -> [b, 4]
            # rewards -> [b, 1]
            # actions -> [b, 1]
            # next_states -> [b, 4]
            # is_overs -> [b, 1]
            states, rewards, actions, next_states, is_overs = self.sample_data(trajectory)
            # 计算values和targets
            # [b, 4] -> [b, 1]
            values = self.model_td.forward(states)
            # [b, 4] -> [b, 1]
            targets = self.model_td.forward(next_states).detach()
            targets = targets * 0.98
            targets *= (1 - is_overs)
            targets += rewards
            # 计算优势,这里的advantages有点像是策略梯度里的reward_sum
            # 只是这里计算的不是reward,而是target和value的差
            # [b, 1]
            deltas = (targets - values).squeeze(dim=1).tolist()
            advantages = self.calc_advantage(deltas)
            advantages = torch.FloatTensor(advantages).reshape(-1, 1)
            # 取出每一步动作的概率
            # [b, 2] -> [b, 2] -> [b, 1]
            old_probs = self.model.forward(states)
            old_probs = old_probs.gather(dim=1, index=actions)
            old_probs = old_probs.detach()
            # 每批数据反复训练10次
            for _ in range(10):
                # 重新计算每一步动作的概率
                # [b, 4] -> [b, 2]
                new_probs = self.model.forward(states)
                # [b, 2] -> [b, 1]
                new_probs = new_probs.gather(dim=1, index=actions)
                new_probs = new_probs
                # 求出概率的变化
                # [b, 1] - [b, 1] -> [b, 1]
                ratios = new_probs / old_probs
                # 计算截断的和不截断的两份loss,取其中小的
                # [b, 1] * [b, 1] -> [b, 1]
                surr1 = ratios * advantages
                # [b, 1] * [b, 1] -> [b, 1]
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                loss = -torch.min(surr1, surr2)
                loss = loss.mean()
                # 重新计算value,并计算时序差分loss
                values = self.model_td.forward(states)
                loss_td = loss_fn(values, targets)
                # 更新参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_td.zero_grad()
                loss_td.backward()
                optimizer_td.step()
            if (epoch + 1) % 50 == 0:
                test_result = sum([self.test(play=False) for _ in range(10)]) / 10
                print(epoch + 1, test_result)


if __name__ == '__main__':
    cart_pole = CartPoleBasedDQN()
    cart_pole.train()
    cart_pole.test(play=False)
    pass
