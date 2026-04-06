#!/bin/bash
#SBATCH --job-name=olmo3-32b-sft
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --mem=0
#SBATCH --output=logs/olmo3-32b-sft-%j.out
#SBATCH --error=logs/olmo3-32b-sft-%j.err

set -euo pipefail

# ---- Paths ----
REPO_DIR=/workspace-vast/wenj/OLMo-core
PYTHON=/workspace-vast/wenj/miniconda3/bin/python
CHECKPOINT=/workspace-vast/wenj/checkpoints/OLMo-3-32B-stage1-step540000-olmocore/model_and_optim
DATASET_PATH=/workspace-vast/wenj/data/dolci-instruct-100k
RUN_NAME="${1:-olmo3-32b-sft-instruct-100k}"

# ---- Environment ----
export HF_HOME=/workspace-vast/pretrained_ckpts
export OLMO_ROOT_DIR=/workspace-vast
export OLMO_SHARED_FS=1
export OLMO_RICH_LOGGING=1
export OMP_NUM_THREADS=8
export FORCE_COLOR=1
export NCCL_DEBUG=WARN
export WANDB_API_KEY=86315b0c394fa72f3907dc58cbb45b6730f193c8

# Distributed training env
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((60000 + SLURM_JOB_ID % 5000))

echo "=== OLMo 3 32B SFT ==="
echo "Run name:   $RUN_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET_PATH"
echo "Node:       $(hostname)"
echo "GPUs:       $SLURM_GPUS_ON_NODE"
echo "Master:     $MASTER_ADDR:$MASTER_PORT"

cd "$REPO_DIR"
mkdir -p logs

srun --kill-on-bad-exit=1 \
    $PYTHON -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$SLURM_NODEID \
    src/scripts/train/sft/Olmo-3-32B-SFT.py train \
        "$RUN_NAME" \
        "$CHECKPOINT" \
        slurm \
        --seq_len=8192 \
        --num_nodes=1 \
        --gpus_per_node=4 \
        --global_batch_size=$((64 * 8192)) \
        --dataset_path="$DATASET_PATH" \
        --no_save_tokenizer
