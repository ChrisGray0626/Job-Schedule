# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/8/1
"""
import numpy as np

from Code import Constant


def execute(current_time, task_type, tasks):
    tasks['current_time'] = current_time
    task_num = len(tasks)
    # Completion ratio
    completed_tasks = tasks[(tasks['completed_time'] != -1) & (tasks['completed_time'] <= current_time)]
    completion_ratio = len(completed_tasks) / task_num
    # Tardiness ratio
    tasks['completed_time'] = tasks['completed_time'].replace(-1, Constant.MAX_VALUE)
    tardiness_tasks = tasks[tasks[['current_time', 'completed_time']].min(axis=1) > tasks['due_time']]
    tardiness_ratio = len(tardiness_tasks) / task_num
    # Waiting time
    mean_waiting_time = current_time - tasks['release_time'].mean()
    waiting_tasks = tasks[tasks['start_time'] == -1]
    # Release time
    minimum_release_time = waiting_tasks['release_time'].min()
    maximum_release_time = waiting_tasks['release_time'].max()
    mean_release_time = waiting_tasks['release_time'].mean()
    # Processing time
    minimum_processing_time = waiting_tasks['processing_time'].min()
    maximum_processing_time = waiting_tasks['processing_time'].max()
    mean_processing_time = waiting_tasks['processing_time'].mean()
    # Remaining processing time
    minimum_remaining_processing_time = waiting_tasks['remaining_processing_time'].min()
    maximum_remaining_processing_time = waiting_tasks['remaining_processing_time'].max()
    mean_remaining_processing_time = waiting_tasks['remaining_processing_time'].mean()
    mean_total_processing_time = waiting_tasks['total_processing_time'].mean()
    # Remaining task num
    minimum_remaining_task_num = waiting_tasks['remaining_task_num'].min()
    maximum_remaining_task_num = waiting_tasks['remaining_task_num'].max()
    mean_remaining_task_num = waiting_tasks['remaining_task_num'].mean()
    # Job release time
    minimum_job_release_time = waiting_tasks['job_release_time'].min()
    mean_job_release_time = waiting_tasks['job_release_time'].mean()
    # Due time
    minimum_due_time = waiting_tasks['due_time'].min()
    mean_due_time = waiting_tasks['due_time'].mean()
    # Stack time
    waiting_tasks['slack_time'] = waiting_tasks['due_time'] - current_time - waiting_tasks[
        'remaining_processing_time']
    minimum_stack_time = waiting_tasks['slack_time'].min()
    mean_stack_time = waiting_tasks['slack_time'].mean()
    # Critical ratio
    waiting_tasks['critical_ratio'] = (waiting_tasks['due_time'] - current_time) / waiting_tasks[
        'total_processing_time']
    minimum_critical_ratio = waiting_tasks['critical_ratio'].min()
    mean_critical_ratio = waiting_tasks['critical_ratio'].mean()

    return np.array(
        [task_type, task_num, mean_job_release_time, mean_processing_time, mean_remaining_processing_time,
         mean_remaining_task_num, mean_due_time, mean_stack_time, mean_waiting_time]
        # [completion_ratio, tardiness_ratio]
        # + [mean_remaining_processing_time / mean_total_processing_time]
        # + [mean_remaining_task_num / 5]
        # + [mean_critical_ratio]
    )


if __name__ == '__main__':
    pass
