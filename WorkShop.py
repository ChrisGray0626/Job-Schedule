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

pd.options.mode.chained_assignment = None


class WorkShop:

    def __init__(self, instance_specification, instance_path, parallel_machine_num, task_schedule_strategy):
        self.instance_specification = instance_specification
        self.instance_path = instance_path
        self.job_type_num = None
        self.job_task_num = None
        self.job_tasks = None
        self.next_task_mat = None
        self.task_processing_times = None
        self.work_centre_num = None
        self.parallel_machine_num = parallel_machine_num
        self.machines = None
        self.jobs = None
        self.tasks = None
        self.task_scheduler = TaskScheduer.get_instance(task_schedule_strategy)
        self.init()

    # def __init__(self):
    #     self.instance_specification = "Taillard"
    #     self.instance_path = "Data/Sample-Taillard.txt"
    #     self.job_type_num = None
    #     self.job_task_num = None
    #     self.job_tasks = None
    #     self.next_task_mat = None
    #     self.task_processing_times = None
    #     self.work_centre_num = None
    #     self.parallel_machine_num = 3
    #     self.machines = None
    #     self.jobs = None
    #     self.tasks = None
    #     self.task_scheduler = TaskScheduer.FIFO()
    #     self.init()

    def init(self):
        self.init_definition()
        self.init_machine()
        self.init_job()
        self.init_task()

    def init_definition(self):
        self.job_type_num, self.job_task_num, self.job_tasks, self.next_task_mat, self.task_processing_times = Util.parse_definition(
            self.instance_specification, self.instance_path)

    def init_machine(self):
        self.machines = pd.DataFrame(columns=['machine_id', 'work_centre_id', 'next_idle_time']).astype(int)
        self.work_centre_num = self.job_task_num
        for i in range(0, self.work_centre_num):
            for j in range(0, self.parallel_machine_num):
                machine_id = len(self.machines)
                self.machines.loc[machine_id] = [machine_id, i, 0]

    def init_job(self):
        self.jobs = pd.DataFrame(
            columns=['job_id', 'job_type', 'create_time', 'start_time', 'completed_time', 'current_task_type',
                     'status']).astype(int)

    def init_task(self):
        self.tasks = pd.DataFrame(
            columns=['task_id', 'job_id', 'task_type', 'processing_time', 'create_time', 'start_time', 'completed_time',
                     'status']).astype(int)

    def add_job(self, job_type, current_time):
        job_id = len(self.jobs)
        create_time = current_time
        start_time = -1
        completed_time = -1
        current_task_type = self.job_tasks[job_type][0]
        status = 0
        job = [job_id, job_type, create_time, start_time, completed_time, current_task_type, status]
        self.jobs.loc[job_id] = job
        self.add_task(job_id, job_type, current_task_type, current_time)

    def add_task(self, job_id, job_type, task_type, current_time):
        task_id = len(self.tasks)
        processing_time = self.task_processing_times[job_type][task_type]
        create_time = current_time
        start_time = -1
        completed_time = -1
        status = 0
        task = [task_id, job_id, task_type, processing_time, create_time, start_time, completed_time, status]
        self.tasks.loc[task_id] = task
        self.task_scheduler.add(task)

    def init_random_job(self, job_num):
        current_times = np.full(self.job_type_num, 0)
        for i in range(0, job_num):
            for j in range(0, self.job_type_num):
                self.add_job(j, current_times[j])
                current_times[j] += np.random.randint(1, 2)

    def schedule(self):
        initial_time = 0
        current_times = np.full(self.work_centre_num, initial_time)
        while not self.task_scheduler.is_empty():
            task_id = self.task_scheduler.peek()
            task = self.tasks.loc[task_id]
            task_type = task['task_type']
            create_time = task['create_time']
            current_time = max(current_times[task_type], create_time)
            # Check if there is idle machine
            idle_machines = self.find_idle_machine(task_type, current_time)
            if len(idle_machines) < 1:
                # There is no idle machine
                continue
            self.task_scheduler.pop()
            # Processing Time
            processing_time = task['processing_time']
            completed_time = current_time + processing_time
            # Choose a machine
            machine = self.choose_machine(idle_machines)
            machine_id = machine['machine_id']
            # Update the next idle time of the machine
            machine['next_idle_time'] = completed_time
            self.machines.loc[machine_id] = machine
            task['start_time'] = current_time
            task['completed_time'] = completed_time
            task['status'] = 1
            job_id = task['job_id']
            job = self.jobs.loc[job_id]
            status = job.loc['status']
            job_type = job.loc['job_type']
            # Update the start time of the job
            if status == 0:
                # Start the job
                job['start_time'] = current_time
                job['status'] = 2
            next_task_type = self.next_task_mat[job_type][task_type]
            # Update the current task of the job
            job['current_task_type'] = next_task_type
            # Update the completed time of the job
            if next_task_type == -1:
                job['completed_time'] = completed_time
                job['status'] = 1
            else:
                # Add the next task
                self.add_task(job_id, job_type, next_task_type, current_time)
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
    work_shop = WorkShop("Taillard", "Data/Sample-Taillard.txt", 3, "FIFO")
    work_shop.init_random_job(50)
    work_shop.schedule()
    work_shop.print()
    pass
