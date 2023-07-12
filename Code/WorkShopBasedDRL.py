# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/9
"""
import random

import numpy as np
import torch

import Constant
from Code.WorkShop import WorkShop
from TaskScheduler import DynamicTaskScheduler
from WorkShopSolution import WorkShopSolution


class WorkShopBasedDRL(WorkShopSolution):

    def __init__(self, _work_shop):
        super().__init__(_work_shop, Constant.DYNAMICAL_SCHEDULING_STRATEGY)
        self.task_scheduler = DynamicTaskScheduler(self.work_shop.task_type)
        self.input_size = 3
        self.output_size = len(Constant.CLASSICAL_SCHEDULING_STRATEGIES)
        self.policy_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output_size),
            torch.nn.Softmax(dim=1),
        )

        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

        self.value_model.forward(torch.randn(2, self.input_size))
        self.policy_model.forward(torch.randn(2, self.input_size))

    def choose_action(self, state):
        state = torch.FloatTensor(state).reshape(1, self.input_size)
        prob = self.policy_model.forward(state)
        # 根据概率选择一个动作
        action = random.choices(range(self.output_size), weights=prob[0].tolist(), k=1)[0]

        return action

    def schedule(self, current_time, task_type, machine_id, is_print=False):
        task_id, state, reward, strategy, next_state, is_over = self.task_scheduler.execute(current_time, task_type,
                                                                                            self.choose_action)
        job_id, task_type = self.work_shop.process(current_time, task_type, machine_id, task_id)
        if is_print:
            print(current_time, job_id, task_type)

        return [state, reward, strategy, next_state, is_over]

    def train(self, epoch_num):
        optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-3)
        optimizer_td = torch.optim.Adam(self.value_model.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(epoch_num):
            # 玩一局游戏,得到数据
            trajectory = self.execute()
            states, rewards, actions, next_states, is_overs = self.parse_trajectory(trajectory)

            values = self.value_model.forward(states)
            # [b, 4] -> [b, 1]
            targets = self.value_model.forward(next_states).detach()
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
                values = self.value_model.forward(states)
                loss_td = loss_fn(values, targets)
                # 更新参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_td.zero_grad()
                loss_td.backward()
                optimizer_td.step()

            if (epoch + 1) % (epoch_num / 10) == 0:
                reward_sum = self.test()
                print(epoch + 1, reward_sum)

    def test(self, is_print=False):
        trajectory = self.execute(is_print)
        reward_sum = sum([i[1] for i in trajectory])

        if is_print:
            self.print()

        return reward_sum

    def parse_trajectory(self, trajectory):
        # [b, 3]
        states = np.array([i[0] for i in trajectory])
        states = torch.FloatTensor(states).reshape(-1, self.input_size)
        # [b, 1]
        rewards = np.array([i[1] for i in trajectory])
        rewards = torch.FloatTensor(rewards).reshape(-1, 1)
        # [b, 1]
        actions = np.array([i[2] for i in trajectory])
        actions = torch.LongTensor(actions).reshape(-1, 1)
        # [b, 3]
        next_states = np.array([i[3] for i in trajectory])
        next_states = torch.FloatTensor(next_states).reshape(-1, self.input_size)
        # [b, 1]
        is_overs = np.array([i[4] for i in trajectory])
        is_overs = torch.LongTensor(is_overs).reshape(-1, 1)

        return states, rewards, actions, next_states, is_overs

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

if __name__ == '__main__':
    instance_specification = "Taillard"
    instance_path = "../Data/Sample-Taillard.txt"
    work_shop = WorkShop(instance_specification, instance_path, 3)
    solution = WorkShopBasedDRL(work_shop)

    solution.train(500)
    solution.test(True)

    pass
