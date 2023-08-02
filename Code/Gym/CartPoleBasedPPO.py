# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/2
"""
import torch

from CartPoleBasedAC import CartPoleBasedAC


class CartPoleBasedPPO(CartPoleBasedAC):

    def __init__(self):
        super().__init__()

    def calc_target(self, rewards, next_states, is_overs):
        with torch.no_grad():
            targets = self.value_model.forward(next_states).detach()
        targets = targets * 0.98
        targets *= (1 - is_overs)
        targets += rewards

        return targets

    @staticmethod
    def calc_advantage(targets, values):
        deltas = (targets - values).squeeze(dim=1).tolist()
        advantages = []
        # 反向遍历 deltas
        s = 0.0
        for delta in deltas[::-1]:
            s = 0.98 * s + delta
            advantages.append(s)
        # 逆序
        advantages.reverse()
        advantages = torch.FloatTensor(advantages).reshape(-1, 1)

        return advantages

    def train(self):
        optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-3)
        optimizer_td = torch.optim.Adam(self.value_model.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()

        # 玩N局游戏,每局游戏训练M次
        for epoch in range(500):
            # 玩一局游戏,得到数据
            # states -> [b, 4]
            # rewards -> [b, 1]
            # actions -> [b, 1]
            # next_states -> [b, 4]
            # overs -> [b, 1]
            trajectory = self.env.play(self.choose_action, False)
            states, rewards, actions, next_states, is_overs = self.parse_trajectory(trajectory)
            # 计算values和targets
            # [b, 4] -> [b, 1]
            values = self.value_model.forward(states)
            # [b, 4] -> [b, 1]
            targets = self.calc_target(rewards, next_states, is_overs)
            # 计算优势,这里的advantages有点像是策略梯度里的reward_sum
            # 只是这里计算的不是reward,而是target和value的差
            # [b, 1]
            advantages = self.calc_advantage(targets, values)
            # 取出每一步动作的概率
            # [b, 2] -> [b, 2] -> [b, 1]
            old_probs = self.policy_model.forward(states)
            old_probs = old_probs.gather(dim=1, index=actions)
            old_probs = old_probs.detach()
            # 每批数据反复训练10次
            for _ in range(10):
                # 重新计算每一步动作的概率
                # [b, 4] -> [b, 2]
                new_probs = self.policy_model.forward(states)
                # [b, 2] -> [b, 1]
                new_probs = new_probs.gather(dim=1, index=actions)
                # 求出概率的变化
                # [b, 1] - [b, 1] -> [b, 1]
                print("new_probs", new_probs)
                print("old_probs", old_probs)
                ratios = new_probs / old_probs
                # 计算截断的和不截断的两份loss,取其中小的
                # [b, 1] * [b, 1] -> [b, 1]
                surr1 = ratios * advantages
                # [b, 1] * [b, 1] -> [b, 1]
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                loss = -torch.min(surr1, surr2)
                loss = loss.mean()
                # 重新计算value,并计算时序差分loss
                values = self.value_model.forward(states)
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
    cart_pole = CartPoleBasedPPO()
    cart_pole.train()
    cart_pole.test(play=False)
    pass
