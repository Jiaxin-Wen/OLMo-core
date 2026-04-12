#!/bin/bash
# Launch two independent OLMo3 7B pre-training runs:
#   Node 1: up_steporder.npy
#   Node 2: down_steporder.npy
# Each runs single-node 8-GPU, no cross-node communication needed.

set -euo pipefail

REPO_DIR=/mnt/vast/jiaxin/OLMo-core
PYTHON=${REPO_DIR}/.venv/bin/python
WORKER_ADDR=91.239.86.229

export WANDB_API_KEY=86315b0c394fa72f3907dc58cbb45b6730f193c8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "=== Launching two independent 7B pre-training runs ==="

# Node 2 (worker): down_steporder
echo "Starting node 2: down_steporder..."
ssh $WORKER_ADDR "cd $REPO_DIR && \
    export WANDB_API_KEY=$WANDB_API_KEY && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    export OMP_NUM_THREADS=8 && \
    $PYTHON -m torch.distributed.run \
        --nproc_per_node=8 \
        --nnodes=1 \
        pretrain_olmo3_7B.py \
            --name olmo3-7b-pretrain-down-steporder \
            --save-folder /mnt/vast/jiaxin/checkpoints/ubuntu/olmo3-7b-pretrain-down-steporder \
            --data-path /mnt/vast/jiaxin/pretrain_data/down_steporder.npy \
    2>&1 | tee logs/pretrain-7b-down-node1-\$(date +%Y%m%d-%H%M%S).log" &
WORKER_PID=$!

# Node 1 (local): up_steporder
echo "Starting node 1: up_steporder..."
$PYTHON -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    pretrain_olmo3_7B.py \
        --name olmo3-7b-pretrain-up-steporder \
        --save-folder /mnt/vast/jiaxin/checkpoints/ubuntu/olmo3-7b-pretrain-up-steporder \
        --data-path /mnt/vast/jiaxin/pretrain_data/up_steporder.npy \
    2>&1 | tee "logs/pretrain-7b-up-node0-$(date +%Y%m%d-%H%M%S).log"
LOCAL_EXIT=$?

wait $WORKER_PID
WORKER_EXIT=$?

echo "Node 1 (up) exit: $LOCAL_EXIT, Node 2 (down) exit: $WORKER_EXIT"
