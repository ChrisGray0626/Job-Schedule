# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/27
"""
from WorkShop import WorkShop

if __name__ == '__main__':
    task_schedule_strategies = ["FIFO", "FILO", "SPT", "LPT", "SRTPT", "LRTPT"]
    instance_specification = "Taillard"
    instance_path = "Data/Sample-Taillard.txt"
    task_schedule_strategy = task_schedule_strategies[5]
    work_shop = WorkShop(instance_specification, instance_path, 3, task_schedule_strategy)
    work_shop.init_random_job(50)
    work_shop.schedule()
    work_shop.print()
    pass
