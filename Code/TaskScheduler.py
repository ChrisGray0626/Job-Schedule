# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""
from queue import PriorityQueue

import pandas as pd

import Constant


# TODO EDD, SS, CR
class ClassicalTaskScheduler:

    def __init__(self, work_centre_num, strategy):
        self.work_centre_num = work_centre_num
        self.strategy = strategy
        self.task_queues = []

        for i in range(work_centre_num):
            self.task_queues.append(TaskQueue.get_instance(strategy))

    def add(self, task_type, item):
        self.task_queues[task_type].add(item)

    def peek(self, task_type):
        return self.task_queues[task_type].peek()

    def poll(self, task_type):
        return self.task_queues[task_type].poll()

    def is_empty(self, task_type):
        return self.task_queues[task_type].is_empty()

    def is_over(self):
        for i in range(self.work_centre_num):
            if not self.is_empty(i):
                return False

        return True

    def size(self, task_type):
        return self.task_queues[task_type].size()


class TaskQueue:

    def __init__(self):
        self.queue = None

    @staticmethod
    def get_instance(strategy):
        if strategy in Constant.CLASSICAL_SCHEDULING_STRATEGIES:
            return ClassicalTaskQueue(strategy)
        elif strategy == Constant.DYNAMICAL_SCHEDULING_STRATEGY:
            return DynamicTaskQueue()
        else:
            raise Exception("Unknown Strategy")

    def add(self, item):
        self.queue.put(item)

    def peek(self):
        return self.queue.queue[0]

    def poll(self):
        if self.queue.empty():
            return -1
        return self.queue.get()

    def is_empty(self):
        return self.queue.empty()

    def size(self):
        return self.queue.qsize()


# Implementation of classical scheduling algorithm
class ClassicalTaskQueue(TaskQueue):

    def __init__(self, strategy):
        super().__init__()
        self.queue = PriorityQueue()
        self.strategy = strategy

    def add(self, item):
        priority = self.calc_priority(self.strategy, item)
        # Job ID as the second priority
        job_id = item[0][0]
        # Task ID
        task_id = item[1][0]
        super().add((priority, job_id, task_id))

    def peek(self):
        # Task ID
        return super().peek()[2]

    def poll(self):
        if super().is_empty():
            return -1
        # Task ID
        return super().poll()[2]

    @staticmethod
    def calc_priority(strategy, item):
        if strategy == 'FIFO':
            # Release Time
            return item[1][4]
        elif strategy == 'FILO':
            # Release Time
            return -item[1][4]
        elif strategy == 'SPT':
            # Processing Time
            return item[1][2]
        elif strategy == 'LPT':
            # Processing Time
            return -item[1][2]
        elif strategy == 'SRTPT':
            # Remaining Processing Time
            return item[0][7]
        elif strategy == 'LRTPT':
            # Remaining Processing Time
            return -item[0][7]
        elif strategy == 'LOR':
            # Remaining Task Num
            return item[0][8]
        elif strategy == 'MOR':
            # Remaining Task Num
            return -item[0][8]
        elif strategy == 'ERD':
            # Release Time of the job
            return item[0][2]
        else:
            raise Exception("Unknown Strategy")


class DynamicTaskScheduler:

    def __init__(self, task_type):
        self.task_type = task_type
        self.task_queues = []

        for i in range(task_type):
            self.task_queues.append(DynamicTaskQueue())

    def add(self, task_type, item):
        self.task_queues[task_type].add(item)

    def execute(self, current_time, task_type, choose_strategy):
        return self.task_queues[task_type].execute(current_time, choose_strategy)

    def is_empty(self, task_type):
        return self.task_queues[task_type].is_empty()

    def is_over(self):
        for i in range(self.task_type):
            if not self.is_empty(i):
                return False

        return True

    def size(self, task_type):
        return self.task_queues[task_type].size()


class DynamicTaskQueue:

    def __init__(self):
        self.queue = pd.DataFrame(
            columns=[
                'task_id', 'job_id', 'task_type', 'processing_time', 'release_time', 'start_time', 'completed_time',
                'status',
                'job_release_time', 'job_start_time', 'job_completed_time', 'remaining_process_time',
                'remaining_task_num'
            ]).astype(int)

    def add(self, item):
        job = item[0]
        job = job[1:]
        job = job.rename(index={'release_time': 'job_release_time', 'start_time': 'job_start_time',
                                'completed_time': 'job_completed_time', 'status': 'job_status'})
        task = item[1]
        task = pd.concat([task, job])
        self.queue.loc[task.loc['task_id']] = task

    def execute(self, current_time, choose_strategy):
        state = self.calc_state(current_time)
        strategy = choose_strategy(state)
        task = self.poll(strategy)
        task_id = task[0]
        process_time = task[3]
        completed_time = current_time + process_time
        next_state = self.calc_state(completed_time)
        reward = self.calc_reward(next_state)
        is_over = self.is_empty()

        return task_id, state, reward, strategy, next_state, is_over

    def calc_state(self, current_time):
        task_num = self.size()
        if task_num == 0:
            return [0, 0, 0]

        mean_waiting_time = current_time - self.queue['release_time'].mean()
        max_waiting_time = current_time - self.queue['release_time'].max()
        mean_remaining_process_time = self.queue['remaining_process_time'].mean()
        max_remaining_process_time = self.queue['remaining_process_time'].max()
        mean_remaining_task_num = self.queue['remaining_task_num'].mean()
        max_remaining_task_num = self.queue['remaining_task_num'].max()

        state = [task_num, mean_waiting_time, max_waiting_time]

        return state

    @staticmethod
    def calc_reward(state):
        task_num = state[0]
        mean_waiting_time = state[1]
        max_waiting_time = state[2]

        reward = - (mean_waiting_time + max_waiting_time)

        return reward

    def poll(self, strategy):
        strategy = Constant.CLASSICAL_SCHEDULING_STRATEGIES[strategy]
        # Sort by job_id when comparison is consistent
        self.queue = self.queue.sort_values('job_id')
        if strategy == 'FIFO':
            task_id = self.queue['release_time'].idxmin()
        elif strategy == 'FILO':
            task_id = self.queue['release_time'].idxmax()
        elif strategy == 'SPT':
            task_id = self.queue['processing_time'].idxmin()
        elif strategy == 'LPT':
            task_id = self.queue['processing_time'].idxmax()
        elif strategy == 'SRTPT':
            task_id = self.queue['remaining_process_time'].idxmin()
        elif strategy == 'LRTPT':
            task_id = self.queue['remaining_process_time'].idxmax()
        elif strategy == 'LOR':
            task_id = self.queue['remaining_task_num'].idxmin()
        elif strategy == 'MOR':
            task_id = self.queue['remaining_task_num'].idxmax()
        elif strategy == 'ERD':
            task_id = self.queue['job_release_time'].idxmin()
        else:
            raise Exception("Unknown Strategy")

        task = self.queue.loc[task_id]
        self.queue.drop(task_id, inplace=True)

        return task

    def is_empty(self):
        return self.queue.empty

    def size(self):
        return self.queue.shape[0]
