# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/23
"""
import numpy as np
import pandas as pd

import Util


class WorkShop:

    def __init__(self):
        self.job_type_num = None
        self.task_type_num = None
        self.job_definitions = None
        self.init_job_definition()
        self.work_centre_num = self.task_type_num
        self.parallel_machine_num = 3
        self.machines = None
        self.init_machine()
        self.jobs = pd.DataFrame(
            columns=['job_type', 'create_time', 'start_time', 'completed_time', 'current_task', 'status'])
        self.tasks = pd.DataFrame(
            columns=['job_id', 'task_type', 'processing_time', 'create_time', 'start_time', 'completed_time', 'status'])

    def init_job_definition(self):
        self.job_type_num, self.task_type_num, job_definitions = Util.parse_job_definition(
            "Data/JobDefinitionSample.txt")
        self.job_definitions = pd.DataFrame(job_definitions,
                                            index=pd.MultiIndex.from_product([range(0, self.job_type_num),
                                                                              ['task_type',
                                                                               'processing_time']])).astype(int)

    def init_machine(self):
        machines = []
        for i in range(0, self.work_centre_num):
            for j in range(0, self.parallel_machine_num):
                machines.append([i, j, 0])
        self.machines = pd.DataFrame(machines, columns=['work_centre_id', 'machine_id', 'status'])

    def add_job(self, current_time):
        job_id = len(self.jobs)
        job_type = np.random.randint(0, self.job_type_num)
        create_time = current_time
        start_time = None
        completed_time = None
        current_task = None
        status = 0
        self.jobs.loc[job_id] = [job_type, create_time, start_time, completed_time, current_task, status]
        self.add_task(job_id, job_type, current_time)

    def add_task(self, job_id, job_type, current_time):
        job_definition = self.job_definitions.loc[job_type]
        task_types = job_definition.loc['task_type']
        processing_times = job_definition.loc['processing_time']
        create_time = current_time
        start_time = None
        completed_time = None
        status = 0
        for i in range(0, self.task_type_num):
            task_id = len(self.tasks)
            task_type = task_types[i]
            processing_time = processing_times[i]
            self.tasks.loc[task_id] = [job_id, task_type, processing_time, create_time, start_time, completed_time,
                                       status]

    def find_task_definition(self, job_type):
        job_definition = self.job_definitions.loc[job_type]
        task_types = job_definition.loc['task_type']
        processing_times = job_definition.loc['processing_time']

        return task_types, processing_times

    def init_random_jobs(self, job_num):
        for i in range(0, job_num):
            self.add_job(0)

    def print(self):
        print(self.jobs)
        print(self.tasks)


if __name__ == '__main__':
    work_shop = WorkShop()
    work_shop.init_random_jobs(10)
    work_shop.print()
    pass
