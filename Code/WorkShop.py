# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/23
"""

import random

import numpy as np
import pandas as pd

import Util

pd.options.mode.chained_assignment = None


class WorkShop:

    def __init__(self, instance_specification, instance_path, parallel_machine_num):
        self.instance_specification = instance_specification
        self.instance_path = instance_path
        self.task_type_num = None
        self.parallel_machine_num = parallel_machine_num
        self.machines = None
        self.job_type_num = None
        self.job_tasks = None
        self.job_first_task = None
        self.jobs = None
        self.next_task_mat = None
        self.processing_times = None
        self.total_processing_times = None
        self.remaining_processing_time_mat = None
        self.tasks = None
        self.operations = None
        self.release = True
        self.init()

    def init(self):
        self.init_definition()
        self.init_machine()
        self.init_job()
        self.init_task()
        self.init_operation()

    def init_definition(self):
        self.job_type_num, self.task_type_num, self.job_tasks, self.processing_times = Util.parse_definition(
            self.instance_specification, self.instance_path)
        self.job_first_task = self.job_tasks[:, 0]
        self.next_task_mat = Util.generate_next_task_mat(self.job_type_num, self.task_type_num, self.job_tasks)
        self.total_processing_times = self.processing_times.sum(axis=1)
        self.remaining_processing_time_mat = Util.generate_remaining_processing_time_mat(self.job_type_num,
                                                                                         self.task_type_num,
                                                                                         self.job_first_task,
                                                                                         self.next_task_mat,
                                                                                         self.processing_times)

    def init_machine(self):
        self.machines = pd.DataFrame(columns=['machine_id', 'work_centre_id', 'next_idle_time']).astype(int)

        for i in range(0, self.task_type_num):
            for j in range(0, self.parallel_machine_num):
                machine_id = len(self.machines)
                self.machines.loc[machine_id] = [machine_id, i, 0]

    def init_job(self):
        self.jobs = pd.DataFrame(
            columns=['job_id', 'job_type', 'release_time', 'start_time', 'completed_time', 'current_task_type',
                     'status', 'remaining_processing_time', 'remaining_task_num', 'total_processing_time',
                     'due_time']).astype(int)

    def init_task(self):
        self.tasks = pd.DataFrame(
            columns=['task_id', 'job_id', 'task_type', 'processing_time', 'release_time', 'start_time',
                     'completed_time',
                     'status']).astype(int)

    def init_operation(self):
        self.operations = pd.DataFrame(
            columns=['operation_id', 'job_type', 'task_type', 'machine_id', 'start_time', 'completed_time']).astype(int)

    def reset(self):
        self.machines['next_idle_time'] = 0
        self.jobs = self.jobs.drop(self.jobs.index)
        self.tasks = self.tasks.drop(self.tasks.index)
        self.operations = self.operations.drop(self.operations.index)
        self.release = True

    @staticmethod
    def calc_due_time(current_time, job_processing_time):
        due_factor_pool = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        due_factor = random.choice(due_factor_pool)
        due_time = int(current_time + job_processing_time * due_factor)

        return due_time

    def add_job(self, job_type, current_time):
        job_id = len(self.jobs)
        release_time = current_time
        start_time = -1
        completed_time = -1
        first_task_type = self.job_first_task[job_type]
        status = 0
        remaining_processing_time = self.total_processing_times[job_type]
        remaining_task_num = self.task_type_num
        total_processing_time = self.total_processing_times[job_type]
        due_time = self.calc_due_time(current_time, self.total_processing_times[job_type])
        # Add the job
        job = [job_id, job_type, release_time, start_time, completed_time, first_task_type, status,
               remaining_processing_time, remaining_task_num, total_processing_time, due_time]
        self.jobs.loc[job_id] = job
        # Add the first task
        first_task_id = self.add_task(job_id, job_type, first_task_type, current_time)

    def add_task(self, job_id, job_type, task_type, current_time):
        task_id = len(self.tasks)
        processing_time = self.processing_times[job_type][task_type]
        release_time = current_time
        start_time = -1
        completed_time = -1
        status = 0
        task = [task_id, job_id, task_type, processing_time, release_time, start_time, completed_time, status]
        self.tasks.loc[task_id] = task

        return task_id

    def init_random_job(self, job_batch_num):
        current_time = 1
        current_times = np.full(self.job_type_num, current_time)
        random.seed(6)
        for i in range(0, job_batch_num):
            for j in range(0, self.job_type_num):
                self.add_job(j, current_times[j])
                current_times[j] += random.randint(0, 5)

        self.release = False

    def process(self, current_time, task_type, machine_id, task_id):
        task = self.tasks.loc[task_id]
        processing_time = task['processing_time']
        completed_time = current_time + processing_time
        # Update the machine
        self.machines.at[machine_id, 'next_idle_time'] = completed_time
        # Update the task
        self.tasks.at[task_id, 'start_time'] = current_time
        self.tasks.at[task_id, 'completed_time'] = completed_time
        self.tasks.at[task_id, 'status'] = 1
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
        self.jobs.at[job_id, 'current_task_type'] = next_task_type
        # Check if the job is completed
        if next_task_type == -1:
            # Update the completed time of the job
            self.jobs.at[job_id, 'completed_time'] = completed_time
            self.jobs.at[job_id, 'status'] = 1
            self.jobs.at[job_id, 'remaining_processing_time'] = 0
        else:
            self.jobs.at[job_id, 'remaining_processing_time'] = self.remaining_processing_time_mat[job_type][
                next_task_type]
            # Add the next task
            next_task_id = self.add_task(job_id, job_type, next_task_type, completed_time)
        self.jobs.at[job_id, 'remaining_task_num'] -= 1

        next_time = self.find_next_idle_time(task_type)

        return job_id, next_time

    def find_idle_machine(self, work_centre_id, current_time):
        machine_ids = self.machines[
            (self.machines['work_centre_id'] == work_centre_id) & (self.machines['next_idle_time'] <= current_time)][
            'machine_id']

        return machine_ids

    def find_next_idle_time(self, work_centre_id):
        next_idle_time = self.machines[self.machines['work_centre_id'] == work_centre_id]['next_idle_time'].min()

        return next_idle_time

    def find_current_task(self, task_type, current_time):
        tasks = self.tasks[(self.tasks['task_type'] == task_type) & (self.tasks['release_time'] <= current_time)]

        return tasks

    def find_pending_task(self, task_type, current_time):
        tasks = self.tasks[(self.tasks['task_type'] == task_type) & (self.tasks['release_time'] <= current_time) & (
                self.tasks['start_time'] == -1)]

        return tasks

    def find_current_job(self, task_type, current_time):
        jobs = self.jobs[(self.jobs['current_task_type'] == task_type) & (self.jobs['release_time'] <= current_time)]

        return jobs

    def find_pending_job(self, task_type, current_time):
        jobs = self.jobs[(self.jobs['current_task_type'] == task_type) & (self.jobs['release_time'] <= current_time) & (
                self.jobs['completed_time'] == -1)]

        return jobs

    def is_over(self):
        pending_job_num = len(self.jobs[self.jobs['status'] != 1])

        return pending_job_num == 0

    def print_result(self):
        job_completed_time = self.jobs['completed_time'].max()
        task_completed_time = self.tasks['completed_time'].max()
        print('Job completed time: ' + str(job_completed_time))
        print('Task completed time: ' + str(task_completed_time))

    @staticmethod
    def merge_task_job(tasks, jobs):
        tasks = tasks.sort_values('job_id')
        jobs = jobs.rename(columns={'release_time': 'job_release_time', 'start_time': 'job_start_time',
                                    'completed_time': 'job_completed_time', 'status': 'job_status'})
        tasks = tasks.merge(jobs, on='job_id', how='left')

        return tasks

    def evaluate_mean_tardiness(self):
        self.jobs['tardiness'] = self.jobs['completed_time'] - self.jobs['due_time']
        self.jobs['tardiness'] = self.jobs['tardiness'].apply(lambda x: max(x, 0))
        mean_tardiness = self.jobs['tardiness'].mean()

        return mean_tardiness
