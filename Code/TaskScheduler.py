# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""


# TODO EDD, SS, CR, Other Strategy
# TODO Clean the code

class ClassicalTaskScheduler:

    @staticmethod
    def execute(strategy, tasks, jobs):
        tasks = tasks.sort_values('job_id')
        jobs = jobs.rename(columns={'release_time': 'job_release_time', 'start_time': 'job_start_time',
                                    'completed_time': 'job_completed_time', 'status': 'job_status'})
        tasks = tasks.merge(jobs, on='job_id', how='left')

        if strategy == 'FIFO':
            idx = tasks['release_time'].idxmin()
        elif strategy == 'FILO':
            idx = tasks['release_time'].idxmax()
        elif strategy == 'SPT':
            idx = tasks['processing_time'].idxmin()
        elif strategy == 'LPT':
            idx = tasks['processing_time'].idxmax()
        elif strategy == 'SRTPT':
            idx = tasks['remaining_process_time'].idxmin()
        elif strategy == 'LRTPT':
            idx = tasks['remaining_process_time'].idxmax()
        elif strategy == 'LOR':
            idx = tasks['remaining_task_num'].idxmin()
        elif strategy == 'MOR':
            idx = tasks['remaining_task_num'].idxmax()
        elif strategy == 'ERD':
            idx = tasks['job_release_time'].idxmin()
        else:
            raise Exception("Unknown Strategy")
        task = tasks.iloc[idx]
        task_id = task['task_id']

        return task_id
