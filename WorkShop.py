# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/23
"""
import numpy as np
import pandas as pd

import TaskScheduer
import Util


class WorkShop:

    def __init__(self):
        self.job_type_num = None
        self.task_type_num = None
        self.job_definitions = None
        self.next_task_mat = None
        self.work_centre_num = None
        self.parallel_machine_num = 3
        self.machines = None
        self.jobs = None
        self.tasks = None
        self.task_scheduler = TaskScheduer.FIFO()
        self.init()

        pd.options.mode.chained_assignment = None

    def init(self):
        self.init_job_definition()
        self.init_next_task_mat()
        self.init_machine()
        self.init_job()
        self.init_task()

    def init_job_definition(self):
        self.job_type_num, self.task_type_num, job_definitions = Util.parse_job_definition(
            "Data/JobDefinitionSample.txt")
        self.job_definitions = pd.DataFrame(job_definitions,
                                            index=pd.MultiIndex.from_product([range(0, self.job_type_num),
                                                                              ['task_type',
                                                                               'processing_time']])).astype(int)
        # Due to the index
        self.job_definitions.loc[(slice(None), 'task_type'), :] -= 1

    def init_next_task_mat(self):
        self.next_task_mat = np.zeros((self.job_type_num, self.task_type_num), dtype=int)
        for i in range(0, self.job_type_num):
            for j in range(0, self.task_type_num - 1):
                self.next_task_mat[i][self.job_definitions.loc[i, 'task_type'].at[j]] = \
                    self.job_definitions.loc[i, 'task_type'].at[j + 1]
            self.next_task_mat[i][self.job_definitions.loc[i, 'task_type'].at[self.task_type_num - 1]] = -1

    def init_machine(self):
        self.machines = pd.DataFrame(columns=['machine_id', 'work_centre_id', 'next_idle_time'])
        self.work_centre_num = self.task_type_num
        for i in range(0, self.work_centre_num):
            for j in range(0, self.parallel_machine_num):
                machine_id = len(self.machines)
                self.machines.loc[machine_id] = [machine_id, int(i), 0]

    def init_job(self):
        self.jobs = pd.DataFrame(
            columns=['job_id', 'job_type', 'create_time', 'start_time', 'completed_time', 'current_task', 'status']).astype(int)

    def init_task(self):
        self.tasks = pd.DataFrame(
            columns=['task_id', 'job_id', 'task_type', 'processing_time', 'create_time', 'start_time', 'completed_time',
                     'status']).astype(int)

    def add_job(self, job_type, current_time):
        job_id = len(self.jobs)
        job_definition = self.job_definitions.loc[job_type]
        create_time = current_time
        start_time = -1
        completed_time = -1
        current_task = job_definition.loc['task_type'][0]
        status = 0
        self.jobs.loc[job_id] = [job_id, job_type, create_time, start_time, completed_time, current_task,
                                 status]
        # self.add_historical_task(job_id, current_time)
        self.add_current_task(job_id, current_time)

    def add_current_task(self, job_id, current_time):
        job = self.jobs.loc[job_id]
        task_type = job.loc['current_task']
        self.add_task(job_id, task_type, current_time)

    def add_task(self, job_id, task_type, current_time):
        task = self.tasks.loc[(self.tasks['job_id'] == job_id) & (self.tasks['task_type'] == task_type)]
        task['create_time'] = current_time
        task = task.squeeze()
        self.task_scheduler.add(task)

    def add_historical_task(self, job_id, current_time):
        job = self.jobs.loc[job_id]
        job_type = job.loc['job_type']
        job_definition = self.job_definitions.loc[job_type]
        task_types = job_definition.loc['task_type']
        processing_times = job_definition.loc['processing_time']
        create_time = current_time
        start_time = -1
        completed_time = -1
        status = 0
        for i in range(0, self.task_type_num):
            task_id = len(self.tasks)
            task_type = task_types[i]
            processing_time = processing_times[i]
            self.tasks.loc[task_id] = \
                [task_id, job_id, task_type, processing_time, create_time, start_time, completed_time, status]
            self.tasks.loc[task_id] = self.tasks.loc[task_id].astype(int)

    def find_task(self, task_id):
        return self.tasks.loc[task_id].copy()

    def find_task_definition(self, job_type):
        job_definition = self.job_definitions.loc[job_type]
        task_types = job_definition.loc['task_type']
        processing_times = job_definition.loc['processing_time']

        return task_types, processing_times

    def init_random_job(self, job_num):
        # TODO 耗时过长，需要优化
        current_times = np.full(self.work_centre_num, 0)
        for i in range(0, job_num):
            for j in range(0, self.job_type_num):
                self.add_job(j, current_times[j])
                current_times[j] += np.random.randint(0, 99)

    def schedule(self):
        initial_time = 0
        current_times = np.full(self.work_centre_num, initial_time)
        while not self.task_scheduler.is_empty():
            task_id = self.task_scheduler.peek()
            task = self.find_task(task_id)
            task_type = task['task_type']
            create_time = task['create_time']
            current_time = max(current_times[task_type], create_time)
            # Check if there is idle machine
            idle_machines = self.find_idle_machine(task_type, current_time)
            if len(idle_machines) < 1:
                # There is no idle machine
                continue
            self.task_scheduler.pop()

            processing_time = task['processing_time']
            completed_time = current_time + processing_time

            # Choose a machine
            machine = self.choose_machine(idle_machines)
            machine_id = machine['machine_id']
            # Update the next idle time of the machine
            machine['next_idle_time'] = completed_time
            self.machines.loc[machine_id] = machine

            task_id = task['task_id']
            # Update the task
            task['start_time'] = current_time
            task['completed_time'] = completed_time
            task['status'] = 1
            self.tasks.loc[task_id] = task

            job_id = task['job_id']
            job = self.jobs.loc[job_id].copy()
            status = job.loc['status']
            job_type = job.loc['job_type']
            # Update the start time of the job
            if status == 0:
                # Start the job
                # Start the job
                job['start_time'] = current_time
                job['status'] = 2
            next_task_type = self.next_task_mat[job_type][task_type]
            # Update the current task of the job
            job['current_task'] = next_task_type

            # Update the completed time of the job
            if next_task_type == -1:
                job['completed_time'] = completed_time
                job['status'] = 1
            else:
                # Add the next task
                self.add_task(job_id, next_task_type, current_time)
            self.jobs.loc[job_id] = job
            # Update the current time
            current_time = self.find_work_centre_machine(task_type)['next_idle_time'].min()
            current_times[task_type] = current_time

    def find_idle_machine(self, work_centre_id, current_time):
        machines = self.machines[
            (self.machines['work_centre_id'] == work_centre_id) & (self.machines['next_idle_time'] <= current_time)]

        return machines

    def find_work_centre_machine(self, work_centre_id):
        machines = self.machines[(self.machines['work_centre_id'] == work_centre_id)]

        return machines

    # TODO Choose Machine
    def choose_machine(self, machines):
        machine_id = machines.index[np.random.randint(0, len(machines))]

        return machines.loc[machine_id].copy()

    def print(self):
        job_completed_time = self.jobs['completed_time'].max()
        task_completed_time = self.tasks['completed_time'].max()
        print('Job completed time: ' + str(job_completed_time))
        print('Task completed time: ' + str(task_completed_time))
        print(self.jobs['completed_time'])
        print(self.tasks['completed_time'])


if __name__ == '__main__':
    # TODO 优化存储结构 DataFrame
    # TODO ID 生成
    work_shop = WorkShop()
    work_shop.init_random_job(100)
    work_shop.schedule()
    work_shop.print()
    pass
