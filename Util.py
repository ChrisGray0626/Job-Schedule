# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/23
"""

import uuid


def parse_job_definition(job_definition_path):
    job_definitions = []
    with open(job_definition_path, 'r') as f:
        lines = f.readlines()
        job_type_num, operation_num = lines[0].replace('\n', '').split(' ')
        job_type_num = int(job_type_num)
        operation_num = int(operation_num)
        for i in range(1, len(lines)):
            job_definitions.append(lines[i].strip().split(' '))

    return job_type_num, operation_num, job_definitions


def generate_uuid():
    return uuid.uuid1().hex


if __name__ == '__main__':
    pass
