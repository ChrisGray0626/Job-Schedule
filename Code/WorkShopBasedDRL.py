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

        # 玩N局游戏,每局游戏训练一次
        for epoch in range(epoch_num):
            # 玩一局游戏,得到数据
            trajectory = self.execute()
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
                state = torch.FloatTensor(states[i]).reshape(1, self.input_size)
                # [1, 3] -> [1, 2]
                prob = self.policy_model.forward(state)
                # [1, 2] -> scala
                prob = prob[0, actions[i]]
                # 根据求导公式,符号取反是因为这里是求loss,所以优化方向相反
                loss = -prob.log() * reward_sum
                # 累积梯度
                loss.backward(retain_graph=True)

            optimizer.step()
            if (epoch + 1) % 5 == 0:
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


if __name__ == '__main__':
    instance_specification = "Taillard"
    instance_path = "../Data/Sample-Taillard.txt"
    work_shop = WorkShop(instance_specification, instance_path, 3)
    solution = WorkShopBasedDRL(work_shop)

    solution.train(50)
    solution.test(True)

    pass
