# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/8/1
"""
from Code import Constant


def execute(current_time, current_tasks, next_time, next_tasks):
    current_tardiness = calc_mean_tardiness(current_time, current_tasks)
    next_tardiness = calc_mean_tardiness(next_time, next_tasks)
    current_waiting_time = calc_mean_waiting_time(current_time, current_tasks)
    next_waiting_time = calc_mean_waiting_time(current_time, next_tasks)
    # reward = 10 / (1 + 1 * current_tardiness)
    reward = 0.1 * (current_tardiness - next_tardiness) + 0.9 * (current_waiting_time - next_waiting_time)

    return reward


# Positive to 1; Negative to 0
def calc_tardiness_value(current_time, tasks):
    tasks['current_time'] = current_time
    tasks['completed_time'] = tasks['completed_time'].replace(-1, Constant.MAX_VALUE)
    tasks['tardiness'] = tasks[['current_time', 'completed_time']].min(axis=1) - tasks['due_time']
    tasks['tardiness'] = tasks['tardiness'].apply(lambda x: 1 if x > 0 else 0)

    return tasks['tardiness'].sum()


def calc_total_tardiness(current_time, tasks):
    tasks['current_time'] = current_time
    tasks['completed_time'] = tasks['completed_time'].replace(-1, Constant.MAX_VALUE)
    tasks['tardiness'] = tasks[['current_time', 'completed_time']].min(axis=1) - tasks['due_time']
    tasks['tardiness'] = tasks['tardiness'].apply(lambda x: max(x, 0))

    return tasks['tardiness'].sum()


def calc_mean_tardiness(current_time, tasks):
    tasks['current_time'] = current_time
    tasks['completed_time'] = tasks['completed_time'].replace(-1, Constant.MAX_VALUE)
    tasks['tardiness'] = tasks[['current_time', 'completed_time']].min(axis=1) - tasks['due_time']
    tasks['tardiness'] = tasks['tardiness'].apply(lambda x: max(x, 0))
    mean_tardiness = tasks['tardiness'].mean()

    return mean_tardiness


def calc_mean_waiting_time(current_time, tasks):
    tasks['current_time'] = current_time
    tasks['waiting_time'] = tasks['current_time'] - tasks['job_release_time']

    mean_waiting_time = tasks['waiting_time'].mean()

    return mean_waiting_time


if __name__ == '__main__':
    pass
