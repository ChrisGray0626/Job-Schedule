# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/2
"""
import random

import torch

from CartPoleSolution import CartPoleSolution


class CartPoleBasedPolicyGradient(CartPoleSolution):

    def __init__(self):
        super().__init__()
        self.policy_model = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Softmax(dim=1),
        )

        self.policy_model.forward(torch.randn(2, 4))

    def choose_action(self, state):
        state = torch.FloatTensor(state).reshape(1, 4)
        # [1, 4] -> [1, 2]
        prob = self.policy_model.forward(state)
        # 根据概率选择一个动作
        action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]

        return action

    def train(self):
        optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-3)

        # 玩N局游戏,每局游戏训练一次
        for epoch in range(1000):
            # 玩一局游戏,得到数据
            trajectory = self.env.play(self.choose_action, False)
            states, rewards, actions, next_states, is_overs = self.parse_trajectory(trajectory)

            optimizer.zero_grad()
            # 反馈的和,初始化为0
            reward_sum = 0
            # 从最后一步算起
            for i in reversed(range(len(states))):
                # 反馈的和,从最后一步的反馈开始计算
                # 每往前一步,>>和<<都衰减0.02,然后再加上当前步的反馈
                reward_sum *= 0.98
                reward_sum += rewards[i]
                # 重新计算对应动作的概率
                state = torch.FloatTensor(states[i]).reshape(1, 4)
                # [1, 4] -> [1, 2]
                prob = self.policy_model.forward(state)
                # [1, 2] -> scala
                prob = prob[0, actions[i]]
                # 根据求导公式,符号取反是因为这里是求loss,所以优化方向相反
                loss = -prob.log() * reward_sum
                # 累积梯度
                loss.backward(retain_graph=True)

            optimizer.step()
            if (epoch + 1) % 100 == 0:
                test_result = sum([self.test(play=False) for _ in range(10)]) / 10
                print(epoch + 1, test_result)


if __name__ == '__main__':
    cart_pole = CartPoleBasedPolicyGradient()
    cart_pole.train()
    cart_pole.test(play=False)
    pass
