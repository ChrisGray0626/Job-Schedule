# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/8
"""
import time

import Constant
from TaskScheduler import ClassicalTaskScheduler
from WorkShop import WorkShop


class WorkShopSolution:
    def __init__(self, _work_shop, _task_schedule_strategy, _job_batch_num=10):
        self.work_shop = _work_shop
        self.task_schedule_strategy = _task_schedule_strategy
        self.job_batch_num = _job_batch_num
        self.task_scheduler = ClassicalTaskScheduler()

    def schedule(self, current_time, task_type, tasks, jobs, machine_id, print_flag=False):
        task_id = self.task_scheduler.execute(self.task_schedule_strategy, tasks, jobs)
        job_id, completed_time = self.work_shop.process(current_time, task_type, machine_id, task_id)

        if print_flag:
            print(current_time, job_id, task_type)

        return current_time, job_id, task_type

    def execute(self, print_flag=False):
        self.work_shop.reset()
        self.work_shop.init_random_job(self.job_batch_num)

        trajectory = [[] for _ in range(self.work_shop.task_type_num)]
        current_time = 1
        while self.work_shop.release or not self.work_shop.is_over():
            for task_type in range(self.work_shop.task_type_num):
                info = self.work_centre_execute(current_time, task_type, print_flag)
                trajectory[task_type] += info
            current_time += 1

        return trajectory

    def work_centre_execute(self, current_time, task_type, print_flag=False):
        trajectory = []
        # Find the idle machines
        machine_ids = self.work_shop.find_idle_machine(task_type, current_time)
        for machine_id in machine_ids:
            tasks = self.work_shop.find_pending_task(task_type, current_time)
            if len(tasks) == 0:
                continue
            if len(tasks) == 1:
                task_id = tasks.index[0]
                job_id, completed_time = self.work_shop.process(current_time, task_type, machine_id, task_id)
                if print_flag:
                    print(current_time, job_id, task_type)
                continue
            jobs = self.work_shop.find_pending_job(task_type, current_time)
            info = self.schedule(current_time, task_type, tasks, jobs, machine_id, print_flag)
            trajectory.append(info)

        return trajectory

    def print_result(self):
        self.work_shop.print_result()


if __name__ == '__main__':
    instance_specification = "Taillard"
    instance_path = "../Data/la01-Taillard.txt"
    task_schedule_strategy = Constant.SCHEDULING_STRATEGIES[0]
    work_shop = WorkShop(instance_specification, instance_path, 3)
    solution = WorkShopSolution(work_shop, task_schedule_strategy)
    start_time = time.time()
    trajectory = solution.execute(print_flag=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", execution_time)
    solution.print_result()
    pass
