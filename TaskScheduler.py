# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""
from queue import PriorityQueue

import Constant


# TODO LOR, MOR, EDD, ERD, SS, CR
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
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))

    def peek(self):
        return super().peek()[1]

    def poll(self):
        return super().poll()[1]

    @staticmethod
    def calc_priority(strategy, task):
        if strategy == 'FIFO':
            # Create Time
            return task[4]
        elif strategy == 'FILO':
            # Create Time
            return -task[4]
        elif strategy == 'SPT':
            # Processing Time
            return task[2]
        elif strategy == 'LPT':
            # Processing Time
            return -task[2]
        elif strategy == 'SRTPT':
            # Remaining Processing Time
            return task[8]
        elif strategy == 'LRTPT':
            # Remaining Processing Time
            return -task[8]
        else:
            raise Exception("Unknown Strategy")
