# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""
from queue import PriorityQueue

import Constant


# TODO EDD, SS, CR
class TaskScheduler:

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

    def is_end(self):
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
        if strategy in Constant.classical_scheduling_strategies:
            return SimpleTaskQueue(strategy)
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


# Implementation of classic scheduling algorithm
class SimpleTaskQueue(TaskQueue):

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
            # Create Time
            return item[1][4] * 1000 + item[0][0]
        elif strategy == 'FILO':
            # Create Time
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
            # Create Time of the job
            return item[0][2]
        else:
            raise Exception("Unknown Strategy")
