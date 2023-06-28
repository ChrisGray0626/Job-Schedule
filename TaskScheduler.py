# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""
from queue import PriorityQueue


# TODO LOR, MOR, EDD, ERD, SS, CR
class TaskScheduler:

    def __init__(self, work_centre_num, strategy):
        self.work_centre_num = work_centre_num
        self.strategy = strategy
        self.task_queues = []

        for i in range(work_centre_num):
            self.task_queues.append(TaskQueue.get_instance(strategy))

    def add(self, task_type, task):
        self.task_queues[task_type].add(task)

    def peek(self, task_type):
        return self.task_queues[task_type].peek()

    def pop(self, task_type):
        return self.task_queues[task_type].pop()

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
        if strategy == 'FIFO':
            return FIFO()
        elif strategy == 'FILO':
            return FILO()
        elif strategy == 'SPT':
            return SPT()
        elif strategy == 'LPT':
            return LPT()
        elif strategy == 'SRTPT':
            return SRTPT()
        elif strategy == 'LRTPT':
            return LRTPT()
        else:
            raise Exception("Unknown Strategy")

    def add(self, item):
        self.queue.put(item)

    def peek(self):
        return self.queue.queue[0]

    def pop(self):
        return self.queue.get()

    def is_empty(self):
        return self.queue.empty()

    def size(self):
        return self.queue.qsize()


class SimpleTaskQueue(TaskQueue):

    def __init__(self):
        super().__init__()
        self.queue = PriorityQueue()

    def peek(self):
        return super().peek()[1]

    def pop(self):
        return super().pop()[1]


class FIFO(SimpleTaskQueue):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # TODO Repeat operation
        # Create Time
        priority = item[4]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


class FILO(SimpleTaskQueue):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Create Time
        priority = -item[4]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


class SPT(SimpleTaskQueue):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Processing Time
        priority = item[2]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


class LPT(SimpleTaskQueue):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Processing Time
        priority = -item[2]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


class SRTPT(SimpleTaskQueue):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Remaining Processing Time
        priority = item[8]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


class LRTPT(SimpleTaskQueue):

    def __init__(self):
        super().__init__()

    def add(self, item):
        # Remaining Processing Time
        priority = -item[8]
        # Task ID
        task_id = item[0]
        super().add((priority, task_id))


if __name__ == '__main__':
    pass
