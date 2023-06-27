# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""
from queue import LifoQueue
from queue import PriorityQueue


def get_instance(strategy):
    if strategy == 'FIFO':
        return FIFO()
    elif strategy == 'FILO':
        return FILO()
    elif strategy == 'SPT':
        return SPT()
    elif strategy == 'LPT':
        return LPT()
    else:
        raise Exception("Unknown Strategy")


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


class PriorityTaskScheduler(TaskScheduler):

    def __init__(self):
        super().__init__()
        self.queue = PriorityQueue()

    def peek(self):
        return super().peek()[1]

    def pop(self):
        return super().pop()[1]


class FIFO(PriorityTaskScheduler):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Create Time
        priority = item[4]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))



class FILO(PriorityTaskScheduler):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Create Time
        priority = -item[4]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


class SPT(PriorityTaskScheduler):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Processing Time
        priority = item[2]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


class LPT(PriorityTaskScheduler):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Processing Time
        priority = -item[2]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


class SRTPT(PriorityTaskScheduler):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Remaining Processing Time
        priority = -item[2]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


if __name__ == '__main__':
    pass
