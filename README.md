# Installation

```bash
pip install -r requirements.txt
```

# Data Structure

## Machine

1. `machine_id`
1. `work_centre_id`
1. `next_idle_time`

## Job

1. `job_id`
1. `job_type`
1. `release_time`
1. `start_time`
1. `completed_time`
1. `current_task_type`
1. `status`
1. `remaining_processing_time`
1. `remaining_task_num`

## Task

1. `task_id`
1. `job_id`
1. `task_type`
1. `processing_time`
1. `release_time`
1. `start_time`
1. `completed_time`
1. `status`


# Task Scheduling Strategy

1. FIFO
1. FILO
1. SPT
1. LPT
1. SRTPT
1. LRTPT
1. LOR
1. MOR
1. ERD

# PyTorch (GPU)

- torch 1.13.0+cu116
- torchaudio 0.13.0
- torchvision 0.14

