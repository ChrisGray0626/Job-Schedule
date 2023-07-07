# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/8
"""
import time

import Constant
from TaskScheduler import TaskScheduler
from WorkShop import WorkShop


class WorkShopSolution:
    def __init__(self, _work_shop, _task_schedule_strategy):
        self.work_shop = _work_shop
        self.task_schedule_strategy = _task_schedule_strategy
        self.task_scheduler = TaskScheduler(self.work_shop.work_centre_num, self.task_schedule_strategy)

    def init_random_job(self, job_num):
        self.work_shop.init_random_job(job_num)

    def schedule(self):
        while self.work_shop.release or not self.work_shop.events.empty():
            current_time, event_type, work_centre_id, param = self.work_shop.events.get(block=True)
            # Machine event
            if event_type == Constant.MACHINE_EVENT:
                machine_id = param
                # There are tasks in the queue need to be processed
                if machine_id == -1:
                    # Find the idle machines
                    machine_ids = self.work_shop.find_idle_machine(work_centre_id, current_time)
                    for machine_id in machine_ids:
                        if not self.task_scheduler.is_empty(work_centre_id):
                            task_id = self.task_scheduler.poll(work_centre_id)
                            self.work_shop.process(current_time, work_centre_id, machine_id, task_id)
                # The machine is idle
                else:
                    # Find a task in the queue
                    if not self.task_scheduler.is_empty(work_centre_id):
                        task_id = self.task_scheduler.poll(work_centre_id)
                        self.work_shop.process(current_time, work_centre_id, machine_id, task_id)
            # Task event
            elif event_type == Constant.TASK_EVENT:
                task_id = param
                task = self.work_shop.tasks.loc[task_id]
                job_id = task['job_id']
                job = self.work_shop.jobs.loc[job_id]
                self.task_scheduler.add(work_centre_id, (job, task))
                # Check if there is the only one task in the queue
                if self.task_scheduler.size(work_centre_id) == 1:
                    # Add the machine event to notify the idle machine
                    self.work_shop.events.put([current_time, Constant.MACHINE_EVENT, work_centre_id, -1])

    def print(self):
        self.work_shop.print()


if __name__ == '__main__':
    instance_specification = "Taillard"
    instance_path = "../Data/Sample-Taillard.txt"
    task_schedule_strategy = Constant.scheduling_strategies[0]
    work_shop = WorkShop(instance_specification, instance_path, 3, task_schedule_strategy)
    solution = WorkShopSolution(work_shop, task_schedule_strategy)

    solution.init_random_job(50)
    start_time = time.time()
    solution.schedule()
    end_time = time.time()
    execution_time = end_time - start_time

    print("Scheduling Execution Time: ", execution_time)
    solution.print()
    pass
