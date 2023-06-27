# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/23
"""

import uuid

import numpy as np


def parse_job_definition(job_definition_path):
    with open(job_definition_path, 'r') as f:
        lines = f.readlines()
        job_type_num, operation_num = lines[0].strip().split(' ')
        job_type_num = int(job_type_num)
        operation_num = int(operation_num)
        job_definitions = []
        for i in range(1, len(lines)):
            job_definitions.append(lines[i].strip().split(' '))

    return job_type_num, operation_num, job_definitions


def parse_definition(instance_specification, file_path):
    if instance_specification == "Standard":
        return parse_standard(file_path)
    if instance_specification == "Taillard":
        return parse_taillard(file_path)


def parse_standard(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        job_type_num, job_task_num = lines[0].strip().split(' ')
        job_type_num = int(job_type_num)
        job_task_num = int(job_task_num)
        job_tasks = np.zeros((job_type_num, job_task_num), dtype=int)
        task_processing_times = np.zeros((job_type_num, job_task_num), dtype=int)
        for i in range(1, job_type_num + 1):
            nums = lines[i].strip().split(' ')
            for j in range(0, job_task_num):
                job_tasks[i - 1][j] = int(nums[j * 2])
                task_processing_times[i - 1][job_tasks[i - 1][j]] = int(nums[j * 2 + 1])
        next_task_mat = parse_next_task_mat(job_type_num, job_task_num, job_tasks)

    return job_type_num, job_task_num, job_tasks, next_task_mat, task_processing_times


def parse_taillard(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        job_type_num, job_task_num = lines[0].strip().split(' ')
        job_type_num = int(job_type_num)
        job_task_num = int(job_task_num)
        job_tasks = np.zeros((job_type_num, job_task_num), dtype=int)
        task_processing_times = np.zeros((job_type_num, job_task_num), dtype=int)
        for i in range(job_type_num + 1, 2 * job_type_num + 1):
            nums = lines[i].strip().split(' ')
            for j in range(0, job_task_num):
                job_type = i - job_type_num - 1
                task_type = int(nums[j]) - 1
                job_tasks[job_type][j] = task_type
        for i in range(1, job_type_num + 1):
            nums = lines[i].strip().split(' ')
            for j in range(0, job_task_num):
                job_type = i - 1
                task_type = job_tasks[job_type][j]
                task_processing_times[job_type][task_type] = int(nums[j])
        next_task_mat = parse_next_task_mat(job_type_num, job_task_num, job_tasks)

    return job_type_num, job_task_num, job_tasks, next_task_mat, task_processing_times


def parse_next_task_mat(job_type_num, job_task_num, job_tasks):
    next_task_mat = np.zeros((job_type_num, job_task_num), dtype=int)
    for i in range(0, job_type_num):
        for j in range(0, job_task_num - 1):
            next_task_mat[i][job_tasks[i][j]] = job_tasks[i][j + 1]
        next_task_mat[i][job_tasks[i][job_task_num - 1]] = -1
    return next_task_mat


def generate_uuid():
    return uuid.uuid1().hex


if __name__ == '__main__':
    pass
