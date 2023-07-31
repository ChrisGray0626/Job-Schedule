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
from Code import Test
from Code.WorkShop import WorkShop
from TaskScheduler import ClassicalTaskScheduler
from WorkShopSolution import WorkShopSolution


# TODO State Reward Status Function
class WorkShopBasedDRL(WorkShopSolution):

    def __init__(self, _work_shop, _job_batch_num=10):
        super().__init__(_work_shop, Constant.DYNAMICAL_SCHEDULING_STRATEGY, _job_batch_num)
        self.task_scheduler = ClassicalTaskScheduler()
        self.job_batch_num = _job_batch_num
        self.input_size = 9
        self.output_size = len(Constant.CLASSICAL_SCHEDULING_STRATEGIES)
        self.device = torch.device("cuda")
        self.policy_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output_size),
            torch.nn.Softmax(dim=1),
        )

        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        random_input = torch.randn(2, self.input_size)

        self.policy_model = self.policy_model.to(self.device)
        self.value_model = self.value_model.to(self.device)
        random_input = random_input.to(self.device)

        # self.value_model.forward(random_input)
        # self.policy_model.forward(random_input)

        self.gamma = 1

    def choose_action(self, state):
        state = torch.FloatTensor(state).reshape(1, self.input_size)
        state = state.to(self.device)

        prob = self.policy_model.forward(state)
        # 根据概率选择一个动作
        action = random.choices(range(self.output_size), weights=prob[0].tolist(), k=1)[0]
        print(prob, action)
        return action

    @staticmethod
    def calc_state(current_time, task_type, tasks):
        tasks['current_time'] = current_time
        task_num = len(tasks)
        # Completion ratio
        completed_tasks = tasks[(tasks['completed_time'] != -1) & (tasks['completed_time'] <= current_time)]
        completion_ratio = len(completed_tasks) / task_num
        # Tardiness ratio
        tasks['completed_time'] = tasks['completed_time'].replace(-1, Constant.MAX_VALUE)
        tardiness_tasks = tasks[tasks[['current_time', 'completed_time']].min(axis=1) > tasks['due_time']]
        tardiness_ratio = len(tardiness_tasks) / task_num
        waiting_tasks = tasks[tasks['start_time'] == -1]
        # Release time
        minimum_release_time = waiting_tasks['release_time'].min()
        maximum_release_time = waiting_tasks['release_time'].max()
        mean_release_time = waiting_tasks['release_time'].mean()
        # Processing time
        minimum_processing_time = waiting_tasks['processing_time'].min()
        maximum_processing_time = waiting_tasks['processing_time'].max()
        mean_processing_time = waiting_tasks['processing_time'].mean()
        # Remaining processing time
        minimum_remaining_processing_time = waiting_tasks['remaining_processing_time'].min()
        maximum_remaining_processing_time = waiting_tasks['remaining_processing_time'].max()
        mean_remaining_processing_time = waiting_tasks['remaining_processing_time'].mean()
        mean_total_processing_time = waiting_tasks['total_processing_time'].mean()
        # Remaining task num
        minimum_remaining_task_num = waiting_tasks['remaining_task_num'].min()
        maximum_remaining_task_num = waiting_tasks['remaining_task_num'].max()
        mean_remaining_task_num = waiting_tasks['remaining_task_num'].mean()
        # Job release time
        minimum_job_release_time = waiting_tasks['job_release_time'].min()
        mean_job_release_time = waiting_tasks['job_release_time'].mean()
        # Due time
        minimum_due_time = waiting_tasks['due_time'].min()
        mean_due_time = waiting_tasks['due_time'].mean()
        # Stack time
        waiting_tasks['slack_time'] = waiting_tasks['due_time'] - current_time - waiting_tasks[
            'remaining_processing_time']
        minimum_stack_time = waiting_tasks['slack_time'].min()
        mean_stack_time = waiting_tasks['slack_time'].mean()
        # Critical ratio
        waiting_tasks['critical_ratio'] = (waiting_tasks['due_time'] - current_time) / waiting_tasks[
            'total_processing_time']
        minimum_critical_ratio = waiting_tasks['critical_ratio'].min()
        mean_critical_ratio = waiting_tasks['critical_ratio'].mean()
        # Waiting time
        mean_waiting_time = current_time - waiting_tasks['release_time'].mean()

        return np.array(
            [task_type, task_num, mean_job_release_time, mean_processing_time, mean_remaining_processing_time, mean_remaining_task_num, mean_due_time, mean_stack_time, mean_waiting_time]
            # [completion_ratio, tardiness_ratio]
            # + [mean_remaining_processing_time / mean_total_processing_time]
            # + [mean_remaining_task_num / 5]
            # + [mean_critical_ratio]
        )

    def calc_reward(self, current_time, current_tasks, next_time, next_tasks):
        current_tardiness = self.calc_mean_tardiness(current_time, current_tasks)
        next_tardiness = self.calc_mean_tardiness(next_time, next_tasks)

        # reward = 10 / (1 + 1 * current_tardiness)
        reward = current_tardiness - next_tardiness

        return reward

    @staticmethod
    def calc_tardiness_value(current_time, tasks):
        tasks['current_time'] = current_time
        tasks['completed_time'] = tasks['completed_time'].replace(-1, Constant.MAX_VALUE)
        tasks['tardiness'] = tasks[['current_time', 'completed_time']].min(axis=1) - tasks['due_time']
        tasks['tardiness'] = tasks['tardiness'].apply(lambda x: 1 if x > 0 else 0)

        return tasks['tardiness'].sum()

    @staticmethod
    def calc_total_tardiness(current_time, tasks):
        tasks['current_time'] = current_time
        tasks['completed_time'] = tasks['completed_time'].replace(-1, Constant.MAX_VALUE)
        tasks['tardiness'] = tasks[['current_time', 'completed_time']].min(axis=1) - tasks['due_time']
        tasks['tardiness'] = tasks['tardiness'].apply(lambda x: max(x, 0))

        return tasks['tardiness'].sum()

    @staticmethod
    def calc_mean_tardiness(current_time, tasks):
        tasks['current_time'] = current_time
        tasks['completed_time'] = tasks['completed_time'].replace(-1, Constant.MAX_VALUE)
        tasks['tardiness'] = tasks[['current_time', 'completed_time']].min(axis=1) - tasks['due_time']
        tasks['tardiness'] = tasks['tardiness'].apply(lambda x: max(x, 0))
        mean_tardiness = tasks['tardiness'].mean()

        return mean_tardiness

    def schedule(self, current_time, task_type, machine_id, print_flag=False):
        tasks = self.work_shop.find_current_task(task_type, current_time)
        jobs = self.work_shop.find_current_job(task_type, current_time)
        tasks = WorkShop.merge_task_job(tasks, jobs)
        # Calculate the state
        state = self.calc_state(current_time, task_type, tasks)
        action = self.choose_action(state)
        strategy = Constant.CLASSICAL_SCHEDULING_STRATEGIES[action]
        waiting_tasks = tasks[tasks['start_time'] == -1]
        task_id = self.task_scheduler.execute(current_time, strategy, waiting_tasks)
        job_id, next_time = self.work_shop.process(current_time, task_type, machine_id, task_id)
        # Calculate the next state
        next_tasks = self.work_shop.find_current_task(task_type, next_time)
        next_jobs = self.work_shop.find_current_job(task_type, next_time)
        next_tasks = WorkShop.merge_task_job(next_tasks, next_jobs)
        next_state = self.calc_state(next_time, task_type, next_tasks)
        # Calculate the reward
        reward = self.calc_reward(current_time, tasks, next_time, next_tasks)
        # Calculate the is_over
        is_over = len(next_tasks) == 1
        print(reward, action)
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
        policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=5e-3)
        value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(epoch_num):
            # 玩一局游戏,得到数据
            trajectory = self.execute()
            for task_type in range(self.work_shop.task_type_num):
                work_centre_trajectory = trajectory[task_type]
                work_centre_trajectory = random.sample(work_centre_trajectory, min(len(work_centre_trajectory), 32))
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
                for _ in range(5):
                    # 重新计算每一步动作的概率
                    # [b, 4] -> [b, 2]
                    new_probs = self.policy_model.forward(states)
                    # [b, 2] -> [b, 1]
                    new_probs = new_probs.gather(dim=1, index=actions)
                    # 求出概率的变化
                    # [b, 1] - [b, 1] -> [b, 1]
                    # print("new_probs", new_probs)
                    ratios = new_probs / old_probs
                    # 计算截断的和不截断的两份loss,取其中小的
                    # [b, 1] * [b, 1] -> [b, 1]
                    surr1 = ratios * advantages
                    # [b, 1] * [b, 1] -> [b, 1]
                    print("new_probs", new_probs)
                    surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                    policy_loss = -torch.min(surr1, surr2)
                    policy_loss = policy_loss.mean()
                    # 重新计算value,并计算时序差分loss
                    values = self.value_model.forward(states)
                    value_loss = loss_fn(values, targets)
                    # 更新参数
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()

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
    Test.torch_test()
    instance_specification = "Taillard"
    instance_path = "../Data/la01-Taillard.txt"
    work_shop = WorkShop(instance_specification, instance_path, 3)
    solution = WorkShopBasedDRL(work_shop, 10)

    solution.train(10)
    solution.test(print_flag=True)

    # solution.run(print_flag=True)
    pass
