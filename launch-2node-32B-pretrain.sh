#!/bin/bash
# Launch OLMo3 32B pre-training on 2x H100 nodes (no SLURM)
# Data is NOT shuffled — preserves exact order from .npy file
#
# Usage:
#   bash launch-2node-32B-pretrain.sh --launch-all
#   bash launch-2node-32B-pretrain.sh --node_rank=0
#   bash launch-2node-32B-pretrain.sh --node_rank=1

set -euo pipefail

# ---- Cluster layout ----
MASTER_ADDR=91.239.86.228   # SF-Compute-odin5-gpu-014
WORKER_ADDR=91.239.86.229   # SF-Compute-odin5-gpu-015
MASTER_PORT=29500
NNODES=2
GPUS_PER_NODE=8

# ---- Paths ----
REPO_DIR=/mnt/vast/jiaxin/OLMo-core
PYTHON=${REPO_DIR}/.venv/bin/python
SAVE_FOLDER=/mnt/vast/jiaxin/checkpoints/ubuntu/olmo3-32b-pretrain-down-steporder
RUN_NAME="${RUN_NAME:-olmo3-32b-pretrain-down-steporder}"

# ---- Environment ----
export OLMO_ROOT_DIR=/mnt/vast/jiaxin
export OLMO_SHARED_FS=1
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000
export WANDB_API_KEY=86315b0c394fa72f3907dc58cbb45b6730f193c8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Network: Gloo for rendezvous, NCCL over InfiniBand
export GLOO_SOCKET_IFNAME=ens10f0np0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# ---- Parse arguments ----
NODE_RANK=""
LAUNCH_ALL=false

for arg in "$@"; do
    case $arg in
        --node_rank=*)
            NODE_RANK="${arg#*=}"
            ;;
        --launch-all)
            LAUNCH_ALL=true
            ;;
        --run_name=*)
            RUN_NAME="${arg#*=}"
            ;;
        --save_folder=*)
            SAVE_FOLDER="${arg#*=}"
            ;;
    esac
done

# ---- Launch on both nodes via SSH ----
if [ "$LAUNCH_ALL" = true ]; then
    echo "=== Launching on both nodes ==="
    SCRIPT_PATH="$(readlink -f "$0")"

    echo "Starting node 1 (worker) on $WORKER_ADDR ..."
    ssh "$WORKER_ADDR" "cd $REPO_DIR && bash $SCRIPT_PATH --node_rank=1 --run_name=$RUN_NAME --save_folder=$SAVE_FOLDER" &
    WORKER_PID=$!

    echo "Starting node 0 (master) locally ..."
    bash "$SCRIPT_PATH" --node_rank=0 --run_name="$RUN_NAME" --save_folder="$SAVE_FOLDER"
    MASTER_EXIT=$?

    wait $WORKER_PID
    WORKER_EXIT=$?

    echo "Master exit: $MASTER_EXIT, Worker exit: $WORKER_EXIT"
    exit $((MASTER_EXIT + WORKER_EXIT))
fi

# ---- Validate ----
if [ -z "$NODE_RANK" ]; then
    echo "Error: specify --node_rank=0 or --node_rank=1, or use --launch-all"
    exit 1
fi

# ---- Info ----
echo "=== OLMo 3 32B Pre-training (2-node H100, no shuffle) ==="
echo "Run name:    $RUN_NAME"
echo "Save folder: $SAVE_FOLDER"
echo "Node:        $(hostname) (rank $NODE_RANK)"
echo "GPUs:        $GPUS_PER_NODE"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo ""

cd "$REPO_DIR"
mkdir -p logs

# ---- Launch torchrun ----
$PYTHON -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    pretrain_olmo3_32B.py \
        --name "$RUN_NAME" \
        --save-folder "$SAVE_FOLDER" \
    2>&1 | tee "logs/pretrain-32b-node${NODE_RANK}-$(date +%Y%m%d-%H%M%S).log"
