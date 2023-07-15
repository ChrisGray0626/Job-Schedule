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
3. `release_time`
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
5. `release_time`
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

# PyTorch (GPU)

- torch 1.13.0+cu116
- torchaudio 0.13.0
- torchvision 0.14

