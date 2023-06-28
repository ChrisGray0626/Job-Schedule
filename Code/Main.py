# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/27
"""
import time

from Code import Constant
from WorkShop import WorkShop

if __name__ == '__main__':
    instance_specification = "Taillard"
    instance_path = "Data/Sample-Taillard.txt"
    task_schedule_strategy = Constant.scheduling_strategies[0]
    work_shop = WorkShop(instance_specification, instance_path, 3, task_schedule_strategy)
    # Thread(target=work_shop.init_random_job(50)).start()
    # Thread(target=work_shop.schedule).start()
    work_shop.init_random_job(50)
    start_time = time.time()
    work_shop.schedule()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Scheduling Execution Time: ", execution_time)
    work_shop.print()
    pass
