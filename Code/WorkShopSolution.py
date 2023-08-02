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

    def schedule(self, current_time, task_type, machine_id, print_flag=False):
        tasks = self.work_shop.find_pending_task(task_type, current_time)
        jobs = self.work_shop.find_pending_job(task_type, current_time)
        tasks = WorkShop.merge_task_job(tasks, jobs)
        task_id = self.task_scheduler.execute(current_time, self.task_schedule_strategy, tasks)
        job_id, next_time = self.work_shop.process(current_time, task_type, machine_id, task_id)

        if print_flag:
            print(current_time, job_id, task_type)

        return current_time, job_id, task_type

    def execute(self, print_flag=False):
        self.work_shop.reset()
        self.work_shop.release_job(self.job_batch_num)

        info_list = [[] for _ in range(self.work_shop.task_type_num)]
        current_time = 1
        while self.work_shop.release or not self.work_shop.is_over():
            for task_type in range(self.work_shop.task_type_num):
                work_centre_info_list = self.work_centre_execute(current_time, task_type, print_flag)
                info_list[task_type] += work_centre_info_list
            current_time += 1
        for i in range(self.work_shop.task_type_num):
            mean_tardiness = self.work_shop.evaluate_work_centre_tardiness(i)
            info_list[i].append([mean_tardiness])
        return info_list

    def work_centre_execute(self, current_time, task_type, print_flag=False):
        info_list = []
        # Find the idle machines
        machine_ids = self.work_shop.find_idle_machine(task_type, current_time)
        for machine_id in machine_ids:
            tasks = self.work_shop.find_pending_task(task_type, current_time)
            if len(tasks) == 0:
                continue
            if len(tasks) == 1:
                task_id = tasks.index[0]
                job_id, next_time = self.work_shop.process(current_time, task_type, machine_id, task_id)
                if print_flag:
                    print(current_time, job_id, task_type)
                continue
            info = self.schedule(current_time, task_type, machine_id, print_flag)
            info_list.append(info)

        return info_list

    def print_result(self):
        print("Task schedule strategy: ", self.task_schedule_strategy)
        self.work_shop.print_result()


def run(work_shop, task_schedule_strategy, print_flag=False):
    solution = WorkShopSolution(work_shop, task_schedule_strategy)
    start_time = time.time()
    trajectory = solution.execute(print_flag=print_flag)
    end_time = time.time()
    execution_time = end_time - start_time
    solution.print_result()


if __name__ == '__main__':
    instance_specification = "Taillard"
    instance_path = "../Data/la01-Taillard.txt"
    # work_shop = WorkShop(instance_specification, instance_path, 3)
    # task_schedule_strategy = Constant.CLASSICAL_SCHEDULING_STRATEGIES[10]
    # run(work_shop, task_schedule_strategy, print_flag=True)
    for task_schedule_strategy in Constant.CLASSICAL_SCHEDULING_STRATEGIES:
        work_shop = WorkShop(instance_specification, instance_path, 3)
        run(work_shop, task_schedule_strategy, print_flag=False)
    pass
