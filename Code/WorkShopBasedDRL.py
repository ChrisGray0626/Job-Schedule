# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/7/9
"""
import time

import Constant
from Code.WorkShop import WorkShop
from TaskScheduler import DynamicTaskScheduler
from WorkShopSolution import WorkShopSolution


class WorkShopBasedDRL(WorkShopSolution):

    def __init__(self, _work_shop):
        super().__init__(_work_shop, Constant.DYNAMICAL_SCHEDULING_STRATEGY)
        self.task_scheduler = DynamicTaskScheduler(self.work_shop.task_type)
        # TODO DRL Model

    def choose_action(self, state):
        action = 0
        return action

    def schedule(self, current_time, task_type, machine_id):
        task_id, state, reward, strategy, next_state, is_over = self.task_scheduler.execute(current_time, task_type, self.choose_action)
        job_id, task_type = self.work_shop.process(current_time, task_type, machine_id, task_id)

        return [current_time, task_type, state, reward, strategy, next_state, is_over]


if __name__ == '__main__':
    instance_specification = "Taillard"
    instance_path = "../Data/Sample-Taillard.txt"
    work_shop = WorkShop(instance_specification, instance_path, 3)
    solution = WorkShopBasedDRL(work_shop)

    solution.init_random_job(50)
    start_time = time.time()
    solution.execute()
    end_time = time.time()
    execution_time = end_time - start_time

    print("Scheduling Execution Time: ", execution_time)
    solution.print()
    pass
