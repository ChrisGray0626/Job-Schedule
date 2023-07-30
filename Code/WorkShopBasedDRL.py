# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/9
"""
import random
import time

import numpy as np
import torch

import Constant
from Code.WorkShop import WorkShop
from TaskScheduler import ClassicalTaskScheduler
from WorkShopSolution import WorkShopSolution


class WorkShopBasedDRL(WorkShopSolution):

    def __init__(self, _work_shop):
        super().__init__(_work_shop, Constant.DYNAMICAL_SCHEDULING_STRATEGY)
        self.task_scheduler = ClassicalTaskScheduler()
        self.input_size = 6
        self.output_size = len(Constant.CLASSICAL_SCHEDULING_STRATEGIES)
        self.device = torch.device("cuda")
        self.policy_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 2 ** self.input_size - 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2 ** self.input_size - 2, 2 ** self.input_size - 1),
            torch.nn.ReLU(),
            torch.nn.Linear(2 ** self.input_size - 1, 2 ** self.input_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 ** self.input_size, self.output_size),
            torch.nn.Softmax(dim=1),
        )

        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 2 ** self.input_size - 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2 ** self.input_size - 2, 2 ** self.input_size - 1),
            torch.nn.ReLU(),
            torch.nn.Linear(2 ** self.input_size - 1, 2 ** self.input_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 ** self.input_size, 1),
        )
        random_input = torch.randn(2, self.input_size)

        self.policy_model.to(self.device)
        self.value_model.to(self.device)
        random_input = random_input.to(self.device)

        self.value_model.forward(random_input)
        self.policy_model.forward(random_input)

        self.gamma = 0.98

    def choose_action(self, state):
        state = torch.FloatTensor(state).reshape(1, self.input_size)
        state = state.to(self.device)

        prob = self.policy_model.forward(state)
        # 根据概率选择一个动作
        action = random.choices(range(self.output_size), weights=prob[0].tolist(), k=1)[0]

        return action

    # TODO State
    @staticmethod
    def calc_state(current_time, tasks):
        task_num = len(tasks)
        mean_release_time = tasks['release_time'].mean()
        mean_waiting_time = current_time - tasks['release_time'].mean()
        mean_processing_time = tasks['processing_time'].mean()
        mean_remaining_processing_time = tasks['remaining_processing_time'].mean()
        mean_remaining_task_num = tasks['remaining_task_num'].mean()

        return np.array(
            [task_num, mean_release_time, mean_waiting_time, mean_processing_time, mean_remaining_processing_time,
             mean_remaining_task_num])

    # TODO Reward
    def calc_reward(self, current_time, tasks):
        # Calculate the mean of waiting time
        mean_waiting_time = current_time - tasks['release_time'].mean()
        # Calculate the mean of remaining processing time ratio
        mean_remaining_processing_time_ratio = tasks.apply(
            lambda row: row['remaining_processing_time'] / self.work_shop.total_processing_times[row['job_type']],
            axis=1).mean()
        # Calculate the mean of remaining task num ratio
        mean_remaining_task_num_ratio = tasks['remaining_task_num'].mean() / self.work_shop.task_type_num

        reward = -mean_waiting_time * mean_remaining_processing_time_ratio * mean_remaining_task_num_ratio

        return reward

    def schedule(self, current_time, task_type, tasks, machine_id, print_flag=False):
        # Calculate the state
        state = self.calc_state(current_time, tasks)
        action = self.choose_action(state)
        strategy = Constant.CLASSICAL_SCHEDULING_STRATEGIES[action]
        task_id = self.task_scheduler.execute(strategy, tasks)
        job_id, completed_time = self.work_shop.process(current_time, task_type, machine_id, task_id)
        # Calculate the next state
        next_tasks = self.work_shop.find_pending_task(task_type, completed_time)
        next_jobs = self.work_shop.find_pending_job(task_type, completed_time)
        next_tasks = self.merge_task_job(next_tasks, next_jobs)
        next_state = self.calc_state(completed_time, next_tasks)
        # Calculate the reward
        reward = self.calc_reward(completed_time, next_tasks)
        # Calculate the is_over
        is_over = len(next_tasks) == 1

        if print_flag:
            print(current_time, job_id, task_type, strategy)

        return state, reward, action, next_state, is_over

    def calc_target(self, rewards, next_states, is_overs):
        with torch.no_grad():
            targets = self.value_model.forward(next_states).detach()
        targets = self.gamma * targets
        targets *= (1 - is_overs)
        targets += rewards

        return targets

    def calc_advantage(self, targets, values):
        deltas = (targets - values).squeeze(dim=1).tolist()
        advantages = []
        # 反向遍历 deltas
        advantage = 0.0
        for delta in deltas[::-1]:
            advantage = self.gamma * advantage + delta
            advantages.append(advantage)
        # 逆序
        advantages.reverse()
        advantages = torch.FloatTensor(advantages).reshape(-1, 1)

        return advantages

    def train(self, epoch_num):
        optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-3)
        optimizer_td = torch.optim.Adam(self.value_model.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(epoch_num):
            # 玩一局游戏,得到数据
            trajectory = self.execute()
            for task_type in range(self.work_shop.task_type_num):
                work_centre_trajectory = trajectory[task_type]
                # TODO min sample ?
                states, rewards, actions, next_states, is_overs = self.parse_trajectory(work_centre_trajectory)

                values = self.value_model.forward(states)
                # [b, 4] -> [b, 1]
                targets = self.calc_target(rewards, next_states, is_overs)
                # 计算优势,这里的advantages有点像是策略梯度里的reward_sum
                # 只是这里计算的不是reward,而是target和value的差
                # [b, 1]
                advantages = self.calc_advantage(targets, values)
                advantages = advantages.to(self.device)
                # 取出每一步动作的概率
                # [b, 2] -> [b, 2] -> [b, 1]
                old_probs = self.policy_model.forward(states)
                old_probs = old_probs.gather(dim=1, index=actions)
                old_probs = old_probs.detach()
                # 每批数据反复训练10次
                for _ in range(1):
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
        # Save the model
        torch.save(self.policy_model.state_dict(), Constant.POLICY_MODEL_PATH)
        torch.save(self.value_model.state_dict(), Constant.VALUE_MODEL_PATH)

    def test(self, print_flag=False):
        trajectory = self.execute(print_flag)
        reward_sum = 0
        for task_type in range(self.work_shop.task_type_num):
            work_centre_trajectory = trajectory[task_type]
            reward_sum += sum([i[1] for i in work_centre_trajectory])

        if print_flag:
            self.print_result()

        return reward_sum

    def run(self, print_flag=False):
        self.policy_model.load_state_dict(torch.load(Constant.POLICY_MODEL_PATH))
        self.value_model.load_state_dict(torch.load(Constant.VALUE_MODEL_PATH))

        start_time = time.time()
        self.execute(print_flag)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time: ", execution_time)

        self.print_result()

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

        states = states.to(self.device)
        rewards = rewards.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        is_overs = is_overs.to(self.device)

        return states, rewards, actions, next_states, is_overs


if __name__ == '__main__':
    instance_specification = "Taillard"
    instance_path = "../Data/la01-Taillard.txt"
    work_shop = WorkShop(instance_specification, instance_path, 3)
    solution = WorkShopBasedDRL(work_shop)

    solution.train(10)
    solution.test(print_flag=True)

    # solution.run(print_flag=True)
    pass
