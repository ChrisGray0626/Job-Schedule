# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""
import numpy as np


class ClassicalTaskScheduler:

    @staticmethod
    def execute(current_time, strategy, tasks):
        tasks = tasks.sort_values('job_id')

        if strategy == 'FIFO':
            idx = tasks['release_time'].idxmin()
        elif strategy == 'FILO':
            idx = tasks['release_time'].idxmax()
        elif strategy == 'SPT':
            idx = tasks['processing_time'].idxmin()
        elif strategy == 'LPT':
            idx = tasks['processing_time'].idxmax()
        elif strategy == 'SRTPT':
            idx = tasks['remaining_processing_time'].idxmin()
        elif strategy == 'LRTPT':
            idx = tasks['remaining_processing_time'].idxmax()
        elif strategy == 'LOR':
            idx = tasks['remaining_task_num'].idxmin()
        elif strategy == 'MOR':
            idx = tasks['remaining_task_num'].idxmax()
        elif strategy == 'ERD':
            idx = tasks['job_release_time'].idxmin()
        elif strategy == 'EDD':
            idx = tasks['due_time'].idxmin()
        elif strategy == 'SS':
            tasks['slack_time'] = tasks['due_time'] - current_time - tasks['remaining_processing_time']
            idx = tasks['slack_time'].idxmin()
        elif strategy == 'CR':
            tasks['critical_ratio'] = (tasks['due_time'] - current_time) / tasks['total_processing_time']
            idx = tasks['critical_ratio'].idxmin()
        elif strategy == 'RANDOM':
            idx = np.random.choice(tasks.index)
        else:
            raise Exception("Unknown Strategy")
        task = tasks.loc[idx]
        task_id = task['task_id']

        return task_id
