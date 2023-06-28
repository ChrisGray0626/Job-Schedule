# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/23
"""
from queue import PriorityQueue

import numpy as np
import pandas as pd

import TaskScheduler
import Util

pd.options.mode.chained_assignment = None


class WorkShop:

    def __init__(self, instance_specification, instance_path, parallel_machine_num, task_schedule_strategy):
        self.instance_specification = instance_specification
        self.instance_path = instance_path
        self.job_type_num = None
        self.job_task_num = None
        self.job_tasks = None
        self.job_first_task = None
        self.next_task_mat = None
        self.remaining_processing_time_mat = None
        self.task_processing_times = None
        self.work_centre_num = None
        self.parallel_machine_num = parallel_machine_num
        self.machines = None
        self.jobs = None
        self.tasks = None
        self.task_schedule_strategy = task_schedule_strategy
        self.task_scheduler = None
        self.operations = None
        self.events = PriorityQueue()
        self.init()

    def init(self):
        self.init_definition()
        self.init_machine()
        self.init_job()
        self.init_task_scheduler()
        self.init_task()
        self.init_operation()

    def init_definition(self):
        self.job_type_num, self.job_task_num, self.job_tasks, self.task_processing_times = Util.parse_definition(
            self.instance_specification, self.instance_path)
        self.job_first_task = self.job_tasks[:, 0]
        self.next_task_mat = Util.generate_next_task_mat(self.job_type_num, self.job_task_num, self.job_tasks)

        self.remaining_processing_time_mat = Util.generate_remaining_processing_time_mat(self.job_type_num,
                                                                                         self.job_task_num,
                                                                                         self.job_first_task,
                                                                                         self.next_task_mat,
                                                                                         self.task_processing_times)

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
                     'status', 'remaining_process_time']).astype(int)

    def init_task_scheduler(self):
        self.task_scheduler = TaskScheduler.TaskScheduler(self.work_centre_num, self.task_schedule_strategy)

    def init_operation(self):
        self.operations = pd.DataFrame(
            columns=['operation_id', 'job_type', 'task_type', 'machine_id', 'start_time', 'completed_time']).astype(int)

    def add_job(self, job_type, current_time):
        job_id = len(self.jobs)
        create_time = current_time
        start_time = -1
        completed_time = -1
        first_task_type = self.job_first_task[job_type]
        status = 0
        # Add the job
        job = [job_id, job_type, create_time, start_time, completed_time, first_task_type, status]
        self.jobs.loc[job_id] = job
        # Add the first task
        first_task_id = self.add_task(job_id, job_type, first_task_type, current_time)
        # Add the task event
        self.events.put([current_time, 1, first_task_type, first_task_id])

    def add_task(self, job_id, job_type, task_type, current_time):
        task_id = len(self.tasks)
        processing_time = self.task_processing_times[job_type][task_type]
        create_time = current_time
        start_time = -1
        completed_time = -1
        status = 0
        remaining_process_time = self.remaining_processing_time_mat[job_type][task_type]
        task = [task_id, job_id, task_type, processing_time, create_time, start_time, completed_time, status,
                remaining_process_time]
        self.tasks.loc[task_id] = task

        return task_id

    def init_random_job(self, job_num):
        current_times = np.full(self.job_type_num, 0)
        for i in range(0, job_num):
            for j in range(0, self.job_type_num):
                self.add_job(j, current_times[j])
                current_times[j] += np.random.randint(0, 1)

    def process(self, current_time, work_centre_id, machine_id, task_id):
        task = self.tasks.loc[task_id]
        task_type = task['task_type']
        processing_time = task['processing_time']
        completed_time = current_time + processing_time
        # Update the machine
        machine = self.machines.loc[machine_id]
        machine['next_idle_time'] = completed_time
        # Add the machine event
        self.events.put([completed_time, 0, work_centre_id, machine_id])
        # Update the task
        task['start_time'] = current_time
        task['completed_time'] = completed_time
        task['status'] = 1
        # Update the job
        job_id = task['job_id']
        job = self.jobs.loc[job_id]
        status = job.loc['status']
        job_type = job.loc['job_type']
        # Update the start time of the job
        if status == 0:
            # Start the job
            job['start_time'] = current_time
            job['status'] = 2
        # Update the current task of the job
        next_task_type = self.next_task_mat[job_type][task_type]
        job['current_task_type'] = next_task_type
        # Update the completed time of the job
        # Check if the job is completed
        if next_task_type == -1:
            job['completed_time'] = completed_time
            job['status'] = 1
        else:
            # Add the next task
            next_task_id = self.add_task(job_id, job_type, next_task_type, current_time)
            # Add the task event
            self.events.put([completed_time, 1, next_task_type, next_task_id])
        # Update the operation
        operator_id = len(self.operations)
        self.operations.loc[operator_id] = [operator_id, job_type, task_type, machine_id, current_time,
                                            completed_time]

    def schedule(self):
        while not self.events.empty():
            current_time, event_type, work_centre_id, param = self.events.get()
            # Machine event
            if event_type == 0:
                machine_id = param
                # if self.task_scheduler.is_empty(work_centre_id):
                #     continue
                task_id = self.task_scheduler.poll(work_centre_id)
                if task_id == -1:
                    continue
                self.process(current_time, work_centre_id, machine_id, task_id)
            # Task event
            elif event_type == 1:
                task_id = param
                task = self.tasks.loc[task_id]
                self.task_scheduler.add(work_centre_id, task)
                # Check if there is the only one task in the queue
                if self.task_scheduler.size(work_centre_id) == 1:
                    # Choose a idle machine
                    machine_id = self.choose_machine(work_centre_id, current_time)
                    if machine_id == -1:
                        # There is no idle machine
                        continue
                    task_id = self.task_scheduler.poll(work_centre_id)
                    self.process(current_time, work_centre_id, machine_id, task_id)

    def find_idle_machine(self, work_centre_id, current_time):
        machines = self.machines[
            (self.machines['work_centre_id'] == work_centre_id) & (self.machines['next_idle_time'] <= current_time)]

        return machines

    def find_work_centre_machine(self, work_centre_id):
        machines = self.machines[(self.machines['work_centre_id'] == work_centre_id)]

        return machines

    def choose_machine(self, work_centre_id, current_time):
        machines = self.machines[
            (self.machines['work_centre_id'] == work_centre_id) & (self.machines['next_idle_time'] <= current_time)]
        num = len(machines)
        if num < 1:
            return -1
        # Choose a machine with the minimum next idle time
        machine_id = machines.index[np.argmin(machines['next_idle_time'])]

        return machine_id

    def print(self):
        print(self.operations)

        job_completed_time = self.jobs['completed_time'].max()
        task_completed_time = self.tasks['completed_time'].max()
        print('Job completed time: ' + str(job_completed_time))
        print('Task completed time: ' + str(task_completed_time))