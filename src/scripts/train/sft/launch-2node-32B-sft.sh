#!/bin/bash
# Launch OLMo3 32B SFT on 2x H100 nodes (no SLURM)
#
# Usage:
#   On node 0 (master): bash launch-2node-32B-sft.sh --node_rank=0
#   On node 1 (worker): bash launch-2node-32B-sft.sh --node_rank=1
#
# Or launch both from node 0:
#   bash launch-2node-32B-sft.sh --launch-all

set -euo pipefail

# ---- Cluster layout ----
MASTER_ADDR=91.239.86.228   # SF-Compute-odin5-gpu-014
WORKER_ADDR=91.239.86.229   # SF-Compute-odin5-gpu-015
MASTER_PORT=29500
NNODES=2
GPUS_PER_NODE=8

# ---- Paths ----
REPO_DIR=/mnt/vast/jiaxin/OLMo-core
VENV_DIR=${REPO_DIR}/.venv
PYTHON=${VENV_DIR}/bin/python
CHECKPOINT=/mnt/vast/jiaxin/checkpoints/OLMo-3-32B-stage1-step540000-olmocore/model_and_optim
DATASET_PATH=/mnt/vast/jiaxin/data/dolci-instruct-100k
RUN_NAME="${RUN_NAME:-olmo3-32b-sft-instruct-100k}"

# ---- Environment ----
export OLMO_ROOT_DIR=/mnt/vast/jiaxin
export OLMO_SHARED_FS=1
export OLMO_RICH_LOGGING=1
export OMP_NUM_THREADS=8
export FORCE_COLOR=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000  # 30 min timeout instead of 15
export WANDB_API_KEY=86315b0c394fa72f3907dc58cbb45b6730f193c8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_LAUNCH_BLOCKING=1  # only for debugging

# Network: Gloo uses Ethernet for rendezvous, NCCL uses InfiniBand for data
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
        --checkpoint=*)
            CHECKPOINT="${arg#*=}"
            ;;
        --dataset_path=*)
            DATASET_PATH="${arg#*=}"
            ;;
    esac
done

# ---- Launch on both nodes via SSH ----
if [ "$LAUNCH_ALL" = true ]; then
    echo "=== Launching on both nodes ==="
    SCRIPT_PATH="$(readlink -f "$0")"

    # Launch worker (node 1) in background via SSH
    echo "Starting node 1 (worker) on $WORKER_ADDR ..."
    ssh "$WORKER_ADDR" "cd $REPO_DIR && bash $SCRIPT_PATH --node_rank=1 --run_name=$RUN_NAME --checkpoint=$CHECKPOINT --dataset_path=$DATASET_PATH" &
    WORKER_PID=$!

    # Launch master (node 0) locally
    echo "Starting node 0 (master) locally ..."
    bash "$SCRIPT_PATH" --node_rank=0 --run_name="$RUN_NAME" --checkpoint="$CHECKPOINT" --dataset_path="$DATASET_PATH"
    MASTER_EXIT=$?

    # Wait for worker
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

if [ ! -f "$PYTHON" ]; then
    echo "Error: Python not found at $PYTHON. Run venv setup first."
    exit 1
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "Warning: Checkpoint not found at $CHECKPOINT"
    echo "Make sure the checkpoint path is correct before training starts."
fi

# ---- Info ----
echo "=== OLMo 3 32B SFT (2-node H100) ==="
echo "Run name:    $RUN_NAME"
echo "Checkpoint:  $CHECKPOINT"
echo "Dataset:     $DATASET_PATH"
echo "Node:        $(hostname) (rank $NODE_RANK)"
echo "GPUs:        $GPUS_PER_NODE"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "Nodes:       $NNODES"
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
    src/scripts/train/sft/Olmo-3-32B-SFT.py train \
        "$RUN_NAME" \
        "$CHECKPOINT" \
        slurm \
        --seq_len=8192 \
        --num_nodes=$NNODES \
        --gpus_per_node=$GPUS_PER_NODE \
        --global_batch_size=$((64 * 8192)) \
        --dataset_path="$DATASET_PATH" \
        --no_save_tokenizer \
        --train_module.rank_microbatch_size=8192 \
        --train_module.ac_config.mode=full \
        --train_module.dp_config.name=fsdp \
        --train_module.dp_config.shard_degree=-1 \
        --trainer.callbacks.checkpointer.save_interval=123 \
        --trainer.callbacks.checkpointer.ephemeral_save_interval=null \
        --trainer.checkpointer.skip_optim_save=true \
    2>&1 | tee "logs/olmo3-32b-sft-node${NODE_RANK}-$(date +%Y%m%d-%H%M%S).log"
