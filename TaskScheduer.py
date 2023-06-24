# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""
from queue import LifoQueue
from queue import PriorityQueue
from queue import Queue


class TaskScheduler:

    def __init__(self):
        self.queue = None

    def add(self, item):
        self.queue.put(item)

    def peek(self):
        return self.queue.queue[0]

    def pop(self):
        return self.queue.get()

    def is_empty(self):
        return self.queue.empty()


class FIFO(TaskScheduler):

    def __init__(self):
        super().__init__()
        self.queue = PriorityQueue()

    def add(self, item):
        # Create Time
        priority = item['create_time']
        item = item['task_id']
        super().add((int(priority), item))

    def peek(self):
        return super().peek()[1]

    def pop(self):
        return super().pop()[1]


class FILO(TaskScheduler):

    def __init__(self):
        super().__init__()
        self.queue = LifoQueue()


class SPT(TaskScheduler):

    def __init__(self):
        super().__init__()
        self.queue = PriorityQueue()

    def add(self, item):
        # Processing Time
        priority = item[2]
        super().add((priority, item))

    def peek(self):
        return super().peek()[1]

    def pop(self):
        return super().pop()[1]


if __name__ == '__main__':
    pass
