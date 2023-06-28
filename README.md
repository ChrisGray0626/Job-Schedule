# Installation

```bash
pip install -r requirements.txt
```

# Data Structure

## Machine

1. `machine_id`
2. `work_centre_id`
3. `next_idle_time`

## Job

1. `job_id`
2. `job_type`
3. `create_time`
4. `start_time`
5. `completed_time`
6. `current_task_type`
7. `status`
8. `remaining_process_time`
9. `remaining_task_num`

## Task

1. `task_id`
2. `job_id`
3. `task_type`
4. `processing_time`
5. `create_time`
6. `start_time`
7. `completed_time`
8. `status`

## Operation

1. `operation_id`
2. `job_type`
3. `task_type`
4. `machine_id`
5. `start_time`
6. `completed_time`

## Event

1. `current_time`
2. `event_type`
3. `work_centre_id`
4. `machine_id`/`task_id`

### Machine Event

It means that the machine is idle.

`event_type` = 0

### Task Event

It means that the new task is created.

`event_type` = 1

# Task Schedule Strategy

- FIFO
- FILO
- SPT
- LPT
- SRTPT
- LRTPT
- LOR
- MOR
- ERD

# Problem

During initialization, a large number of tasks will be added to the task queue. Since the priority queue is used, it
will be a little time-consuming.