# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/28
"""
import sys

MODEL_PATH = "../Model"
POLICY_MODEL_PATH = MODEL_PATH + "/PolicyModel.pth"
VALUE_MODEL_PATH = MODEL_PATH + "/ValueModel.pth"

CLASSICAL_SCHEDULING_STRATEGIES = ["FIFO", "FILO", "SPT", "LPT", "SRTPT", "LRTPT", "LOR", "MOR", "ERD", "EDD", "SS", "CR"]
DRL_SCHEDULING_STRATEGIES = CLASSICAL_SCHEDULING_STRATEGIES + ["RANDOM"]
DYNAMICAL_SCHEDULING_STRATEGY = "DYNAMIC"

TASK_EVENT = 0
MACHINE_EVENT = 1

MAX_VALUE = int(sys.maxsize)
if __name__ == '__main__':
    pass
