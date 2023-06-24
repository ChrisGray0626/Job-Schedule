# README

## Installation

```bash
pip install -r requirements.txt
```

## Data Structure

### Job

1. `job_id`
2. `job_type`
3. `create_time`
4. `start_time`
5. `completed_time`
6. `current_task_type`
7. `status`

### Task

1. `task_id`
2. `job_id`
3. `task_type`
4. `processing_time`
5. `create_time`
6. `start_time`
7. `completed_time`
8. `status`
9. `remaining_process_time`

## Task Schedule Strategy

1. FIFO
2. FILO
3. SPT
4. LPT
5. SRTPT
6. LRTPT

# Problem

The scheduling of all work centres is carried out in the same scheduler, although this does not affect the results, it will affect the computational efficiency. In the future, the scheduling of each work centre separately will be considered.