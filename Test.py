# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""

import pandas as pd

import Util


def dataframe():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
    print(df)
    a = df.loc[0]
    a['a'] = 10
    print(df)

def parse():
    instance_specification = "Standard"
    file_path = "Data/la01-Standard.txt"
    # file_path = "Data/la01-Taillard.txt"
    job_type_num, job_task_num, job_tasks, next_task_mat, task_processing_times = Util.parse_definition(instance_specification, file_path)
    print(job_tasks)
    print(task_processing_times)

if __name__ == '__main__':
    parse()
    pass
