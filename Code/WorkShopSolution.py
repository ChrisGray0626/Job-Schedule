# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/8
"""
import time

import Constant
from Code import Util
from TaskScheduler import ClassicalTaskScheduler
from WorkShop import WorkShop


class WorkShopSolution:
    def __init__(self, _work_shop, _task_schedule_strategy):
        self.work_shop = _work_shop
        self.task_schedule_strategy = _task_schedule_strategy
        self.task_scheduler = ClassicalTaskScheduler()

    def init_random_job(self, job_num):
        self.work_shop.init_random_job(job_num)

    def schedule(self, current_time, task_type, machine_id, is_print=False):
        task_id = self.task_scheduler.poll(task_type)
        job_id, task_type = self.work_shop.process(current_time, task_type, machine_id, task_id)
        if is_print:
            print(current_time, job_id, task_type)

        return [current_time, job_id, task_type]

    def _execute(self, is_print=False):
        self.work_shop.reset()
        self.init_random_job(300)
        trajectory = []
        while self.work_shop.release or not self.work_shop.events.empty():
            current_time, event_type, task_type, param = self.work_shop.events.get(block=True)
            # Machine event
            if event_type == Constant.MACHINE_EVENT:
                machine_id = param
                # There are tasks in the queue need to be processed
                if machine_id == -1:
                    # Find the idle machines
                    machine_ids = self.work_shop.find_idle_machine(task_type, current_time)
                    for machine_id in machine_ids:
                        if not self.task_scheduler.is_empty(task_type):
                            record = self.schedule(current_time, task_type, machine_id, is_print)
                            trajectory.append(record)
                # The machine is idle
                else:
                    # Find a task in the queue
                    if not self.task_scheduler.is_empty(task_type):
                        record = self.schedule(current_time, task_type, machine_id, is_print)
                        trajectory.append(record)
            # Task event
            elif event_type == Constant.TASK_EVENT:
                task_id = param
                task = self.work_shop.tasks.loc[task_id]
                job_id = task['job_id']
                job = self.work_shop.jobs.loc[job_id]
                self.task_scheduler.add(task_type, (job, task))
                # Check if there is the only one task in the queue
                # if len(self.work_shop.find_work_centre_current_task(task_type, current_time)) == 1:
                # Add the machine event to notify the idle machine
                self.work_shop.events.put([current_time, Constant.MACHINE_EVENT, task_type, -1])
        return trajectory

    def execute(self, is_print=False):
        self.work_shop.reset()
        self.init_random_job(50)
        trajectory = []
        current_time = 1
        while self.work_shop.release or not self.work_shop.is_over():
            for task_type in range(self.work_shop.task_type):
                # Find the idle machines
                machine_ids = self.work_shop.find_idle_machine(task_type, current_time)
                for machine_id in machine_ids:
                    tasks = self.work_shop.find_pending_task(task_type, current_time)
                    if len(tasks) == 0:
                        continue
                    jobs = self.work_shop.find_pending_job(task_type, current_time)
                    task_id = self.task_scheduler.execute(self.task_schedule_strategy, tasks, jobs)
                    job_id, task_type = self.work_shop.process(current_time, task_type, machine_id, task_id)
                    if is_print:
                        trajectory.append([current_time, job_id, task_type])
                        print(current_time, job_id, task_type)
            current_time += 1
        return trajectory

    def print(self):
        self.work_shop.print()


if __name__ == '__main__':
    instance_specification = "Taillard"
    instance_path = "../Data/Sample-Taillard.txt"
    task_schedule_strategy = Constant.SCHEDULING_STRATEGIES[0]
    work_shop = WorkShop(instance_specification, instance_path, 3)
    solution = WorkShopSolution(work_shop, task_schedule_strategy)

    trajectory = solution.execute(is_print=False)
    solution.print()
    pass
