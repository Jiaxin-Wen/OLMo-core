#!/bin/bash
# Resume OLMo3 32B pre-training from an official checkpoint on 2x H100 nodes.
#
# Usage:
#   bash launch-2node-32B-resume-pretrain.sh --launch-all \
#       --load_path=https://olmo-checkpoints.org/ai2-llm/stego32-highlr-filter3/step233000/ \
#       --resume_step=233000 \
#       --max_steps=3000 \
#       --data_path=/mnt/vast/jiaxin/pretrain_data/down_steporder.npy

set -euo pipefail

# ---- Cluster layout ----
MASTER_ADDR=91.239.86.228
WORKER_ADDR=91.239.86.229
MASTER_PORT=29500
NNODES=2
GPUS_PER_NODE=8

# ---- Paths ----
REPO_DIR=/mnt/vast/jiaxin/OLMo-core
PYTHON=${REPO_DIR}/.venv/bin/python

# ---- Defaults ----
LOAD_PATH="https://olmo-checkpoints.org/ai2-llm/stego32-highlr-filter3/step233000/"
RESUME_STEP=233000
MAX_STEPS=3000
DATA_PATH="/mnt/vast/jiaxin/pretrain_data/down_steporder.npy"
SAVE_FOLDER="/mnt/vast/jiaxin/checkpoints/ubuntu/olmo3-32b-resume-step233000"
RUN_NAME="olmo3-32b-resume-step233000"

# ---- Environment ----
export OLMO_ROOT_DIR=/mnt/vast/jiaxin
export OLMO_SHARED_FS=1
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000
export WANDB_API_KEY=86315b0c394fa72f3907dc58cbb45b6730f193c8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME=ens10f0np0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# ---- Parse arguments ----
NODE_RANK=""
LAUNCH_ALL=false

for arg in "$@"; do
    case $arg in
        --node_rank=*) NODE_RANK="${arg#*=}" ;;
        --launch-all) LAUNCH_ALL=true ;;
        --run_name=*) RUN_NAME="${arg#*=}" ;;
        --load_path=*) LOAD_PATH="${arg#*=}" ;;
        --resume_step=*) RESUME_STEP="${arg#*=}" ;;
        --max_steps=*) MAX_STEPS="${arg#*=}" ;;
        --data_path=*) DATA_PATH="${arg#*=}" ;;
        --save_folder=*) SAVE_FOLDER="${arg#*=}" ;;
    esac
done

# ---- Launch on both nodes via SSH ----
if [ "$LAUNCH_ALL" = true ]; then
    echo "=== Launching on both nodes ==="
    SCRIPT_PATH="$(readlink -f "$0")"

    echo "Starting node 1 (worker) on $WORKER_ADDR ..."
    ssh "$WORKER_ADDR" "cd $REPO_DIR && bash $SCRIPT_PATH --node_rank=1 --run_name=$RUN_NAME --load_path=$LOAD_PATH --resume_step=$RESUME_STEP --max_steps=$MAX_STEPS --data_path=$DATA_PATH --save_folder=$SAVE_FOLDER" &
    WORKER_PID=$!

    echo "Starting node 0 (master) locally ..."
    bash "$SCRIPT_PATH" --node_rank=0 --run_name="$RUN_NAME" --load_path="$LOAD_PATH" --resume_step="$RESUME_STEP" --max_steps="$MAX_STEPS" --data_path="$DATA_PATH" --save_folder="$SAVE_FOLDER"
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
echo "=== OLMo 3 32B Resume Pre-training (2-node H100) ==="
echo "Run name:     $RUN_NAME"
echo "Load path:    $LOAD_PATH"
echo "Resume step:  $RESUME_STEP"
echo "Max steps:    $MAX_STEPS"
echo "Data path:    $DATA_PATH"
echo "Save folder:  $SAVE_FOLDER"
echo "Node:         $(hostname) (rank $NODE_RANK)"
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
    resume_pretrain_olmo3_32B.py \
        --name "$RUN_NAME" \
        --save-folder "$SAVE_FOLDER" \
        --data-path "$DATA_PATH" \
        --load-path "$LOAD_PATH" \
        --resume-step "$RESUME_STEP" \
        --max-steps "$MAX_STEPS" \
    2>&1 | tee "logs/resume-32b-node${NODE_RANK}-$(date +%Y%m%d-%H%M%S).log"
