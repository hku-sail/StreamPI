#!/bin/bash

export MASTER_HOST="$VC_WORKER_HOSTS"
export MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
export MASTER_PORT="6060"
export NNODES="${MA_NUM_HOSTS:-1}"
export NODE_RANK="${VC_TASK_INDEX:-0}"
export NGPUS_PER_NODE="$MA_NUM_GPUS"

export WANDB_MODE="offline"

JAX_DEBUG_NOWAIT=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u scripts/train_multi_node.py pi05_calvin_stream5 --exp-name=pi05_stream5_calvin_4nodes --overwrite --batch-size 256 --fsdp-devices=4
