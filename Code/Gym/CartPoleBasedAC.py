# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/2
"""
import torch

from CartPoleBasedPolicyGradient import CartPoleBasedPolicyGradient


class CartPoleBasedAC(CartPoleBasedPolicyGradient):

    def __init__(self):
        super().__init__()
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

        self.model(torch.randn(2, 4)), self.model_td(torch.randn(2, 4))

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer_td = torch.optim.Adam(self.model_td.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()

        # 玩N局游戏,每局游戏训练一次
        for epoch in range(1000):
            # 玩一局游戏,得到数据
            # states -> [b, 4]
            # rewards -> [b, 1]
            # actions -> [b, 1]
            # next_states -> [b, 4]
            # is_overs -> [b, 1]
            trajectory = self.env.play(self.choose_action, False)
            states, rewards, actions, next_states, is_overs = self.parse_trajectory(trajectory)
            # 计算values和targets
            # [b, 4] -> [b ,1]
            values = self.model_td.forward(states)
            # [b, 4] -> [b ,1]
            targets = self.model_td(next_states) * 0.98
            # [b ,1] * [b ,1] -> [b ,1]
            targets *= (1 - is_overs)
            # [b ,1] + [b ,1] -> [b ,1]
            targets += rewards
            # 时序差分误差
            # [b ,1] - [b ,1] -> [b ,1]
            delta = (targets - values).detach()
            # 重新计算对应动作的概率
            # [b, 4] -> [b ,2]
            probs = self.model.forward(states)
            # [b ,2] -> [b ,1]
            probs = probs.gather(dim=1, index=actions)
            # 根据策略梯度算法的导函数实现
            # 只是把公式中的reward_sum替换为了时序差分的误差
            # [b ,1] * [b ,1] -> [b ,1] -> scala
            loss = (-probs.log() * delta).mean()
            # 时序差分的loss就是简单的value和target求mse loss即可
            loss_td = loss_fn(values, targets.detach())
            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_td.zero_grad()
            loss_td.backward()
            optimizer_td.step()

            if (epoch + 1) % 100 == 0:
                test_result = sum([self.test(play=False) for _ in range(10)]) / 10
                print(epoch + 1, test_result)


if __name__ == '__main__':
    cart_pole = CartPoleBasedAC()
    cart_pole.train()
    cart_pole.test(play=False)
    pass
