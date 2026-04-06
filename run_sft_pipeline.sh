#!/bin/bash
# Sequential SFT training and evaluation pipeline for multiple OLMo3 32B stage1 checkpoints
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

REPO_DIR=/mnt/vast/jiaxin/OLMo-core
VENV=$REPO_DIR/.venv/bin
OLMES_ENV=/mnt/vast/jiaxin/miniforge3/envs/olmes/bin
CKPT_BASE=/mnt/vast/jiaxin/checkpoints
DATA_PATH=/mnt/vast/jiaxin/data/dolci-instruct-100k
HF_REPO=allenai/Olmo-3-1125-32B

# Steps to process
STEPS=(574000 573000 500000 493000 538000 496000 502000 504000)

export OLMO_ROOT_DIR=/mnt/vast/jiaxin
export OLMO_SHARED_FS=1
export OLMO_RICH_LOGGING=1
export OMP_NUM_THREADS=8
export FORCE_COLOR=1
export NCCL_DEBUG=WARN
export WANDB_API_KEY=86315b0c394fa72f3907dc58cbb45b6730f193c8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME=ens10f0np0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

MASTER_ADDR=91.239.86.228
WORKER_ADDR=91.239.86.229
MASTER_PORT=29500

cleanup_gpus() {
    echo "[$(date +%H:%M:%S)] Cleaning up GPU processes..."
    pkill -9 -f "Olmo-3-32B-SFT\|oe_eval\|olmes\|vllm" 2>/dev/null || true
    ssh $WORKER_ADDR "pkill -9 -f 'Olmo-3-32B-SFT\|oe_eval\|olmes\|vllm'" 2>/dev/null || true
    sleep 5
    # Kill any remaining GPU processes by checking nvidia-smi
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 $pid 2>/dev/null || true
    done
    for pid in $(ssh $WORKER_ADDR "nvidia-smi --query-compute-apps=pid --format=csv,noheader" 2>/dev/null); do
        ssh $WORKER_ADDR "kill -9 $pid" 2>/dev/null || true
    done
    sleep 3
    echo "[$(date +%H:%M:%S)] GPU cleanup done"
}

for STEP in "${STEPS[@]}"; do
    echo ""
    echo "================================================================"
    echo "=== Processing step ${STEP} ==="
    echo "================================================================"
    echo "Started at: $(date)"

    HF_DIR=$CKPT_BASE/Olmo-3-1125-32B-stage1-step${STEP}-hf
    OLMO_DIR=$CKPT_BASE/OLMo-3-32B-stage1-step${STEP}-olmocore
    SFT_NAME=olmo3-32b-sft-instruct-100k-step${STEP}
    SFT_DIR=$CKPT_BASE/ubuntu/olmo-sft/$SFT_NAME
    HF_SFT_DIR=$SFT_DIR/step246-hf

    # ---- Step 1: Download HF checkpoint ----
    if [ ! -d "$HF_DIR" ]; then
        echo "[$(date +%H:%M:%S)] Downloading HF checkpoint step${STEP}..."
        $VENV/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', revision='stage1-step${STEP}', local_dir='$HF_DIR')
"
    else
        echo "[$(date +%H:%M:%S)] HF checkpoint already exists: $HF_DIR"
    fi

    # ---- Step 2: Convert HF -> OLMo-core format ----
    if [ ! -d "$OLMO_DIR/model_and_optim" ]; then
        echo "[$(date +%H:%M:%S)] Converting to OLMo-core format..."
        CUDA_VISIBLE_DEVICES="" $VENV/python -c "
import torch, logging
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from tempfile import TemporaryDirectory
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.io import join_path
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer.config import TransformerConfig, TransformerBlockConfig
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.hf.checkpoint import load_hf_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

tokenizer_config = TokenizerConfig.dolma2()
model_config = TransformerConfig.olmo3_32B(vocab_size=tokenizer_config.padded_vocab_size())

# Override attention backend for CPU conversion
assert isinstance(model_config.block, TransformerBlockConfig)
attn_config = model_config.block.sequence_mixer
if isinstance(attn_config, AttentionConfig):
    attn_config.backend = AttentionBackendName.torch
    attn_config.use_flash = False

model = model_config.build(init_device='meta')
model.to_empty(device=torch.device('cpu'))
state_dict_options = dist_cp_sd.StateDictOptions(flatten_optimizer_state_dict=True, cpu_offload=True)
model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

with TemporaryDirectory() as work_dir:
    load_hf_model('$HF_DIR', model_state_dict, work_dir=work_dir, num_embeddings=model.vocab_size)

model.load_state_dict(model_state_dict)
save_model_and_optim_state(join_path('$OLMO_DIR', 'model_and_optim'), model, save_overwrite=True)
log.info('Conversion complete!')
"
    else
        echo "[$(date +%H:%M:%S)] OLMo-core checkpoint already exists: $OLMO_DIR"
    fi

    # ---- Step 3: SFT Training ----
    if [ -d "$SFT_DIR/step246/model_and_optim" ]; then
        echo "[$(date +%H:%M:%S)] SFT checkpoint already exists, skipping training"
    else
        echo "[$(date +%H:%M:%S)] Starting SFT training..."

        # Clean up GPU memory and previous SFT save folder
        cleanup_gpus
        rm -rf "$SFT_DIR" 2>/dev/null
        rm -rf $CKPT_BASE/ubuntu/dataset-cache/dataset-common/ 2>/dev/null

    # Launch on node 1 in background
    ssh $WORKER_ADDR "\
        OLMO_ROOT_DIR=/mnt/vast/jiaxin \
        OLMO_SHARED_FS=1 OLMO_RICH_LOGGING=1 OMP_NUM_THREADS=8 FORCE_COLOR=1 \
        NCCL_DEBUG=WARN WANDB_API_KEY=86315b0c394fa72f3907dc58cbb45b6730f193c8 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        GLOO_SOCKET_IFNAME=ens10f0np0 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=5 \
        $VENV/python -m torch.distributed.run \
            --nnodes=2 --nproc_per_node=8 \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            --node_rank=1 \
            $REPO_DIR/src/scripts/train/sft/Olmo-3-32B-SFT.py train \
                $SFT_NAME \
                $OLMO_DIR/model_and_optim \
                slurm \
                --seq_len=8192 --num_nodes=2 --gpus_per_node=8 \
                --global_batch_size=524288 \
                --dataset_path=$DATA_PATH \
                --no_save_tokenizer \
                --train_module.rank_microbatch_size=8192 \
                --train_module.ac_config.mode=full \
                --train_module.dp_config.name=fsdp \
                --train_module.dp_config.shard_degree=-1 \
                --trainer.callbacks.checkpointer.save_interval=123 \
                --trainer.checkpointer.skip_optim_save=true \
        " > /tmp/sft_node1_step${STEP}.log 2>&1 &
    NODE1_PID=$!

    # Launch on node 0
    $VENV/python -m torch.distributed.run \
        --nnodes=2 --nproc_per_node=8 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        --node_rank=0 \
        $REPO_DIR/src/scripts/train/sft/Olmo-3-32B-SFT.py train \
            $SFT_NAME \
            $OLMO_DIR/model_and_optim \
            slurm \
            --seq_len=8192 --num_nodes=2 --gpus_per_node=8 \
            --global_batch_size=524288 \
            --dataset_path=$DATA_PATH \
            --no_save_tokenizer \
            --train_module.rank_microbatch_size=8192 \
            --train_module.ac_config.mode=full \
            --train_module.dp_config.name=fsdp \
            --train_module.dp_config.shard_degree=-1 \
            --trainer.callbacks.checkpointer.save_interval=123 \
            --trainer.checkpointer.skip_optim_save=true \
        2>&1 | tee /tmp/sft_node0_step${STEP}.log

    wait $NODE1_PID || true
    echo "[$(date +%H:%M:%S)] SFT training complete for step${STEP}"
    fi

    # ---- Step 4: Convert SFT checkpoint to HF ----
    if [ -d "$HF_SFT_DIR" ] && [ "$(ls $HF_SFT_DIR/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] HF checkpoint already exists, skipping conversion"
    else
        echo "[$(date +%H:%M:%S)] Converting SFT checkpoint to HF format..."
        CUDA_VISIBLE_DEVICES="" $VENV/python \
            $REPO_DIR/src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i $SFT_DIR/step246 \
            -o $HF_SFT_DIR \
            -t allenai/dolma2-tokenizer \
            --skip-validation
        echo "[$(date +%H:%M:%S)] HF conversion complete"
    fi

    # ---- Step 5: Evaluate ----
    echo "[$(date +%H:%M:%S)] Starting evaluation..."

    EVAL_DIR=/mnt/vast/jiaxin/eval-results-step${STEP}
    mkdir -p $EVAL_DIR

    # Retry eval up to 2 times
    for attempt in 1 2; do
        echo "[$(date +%H:%M:%S)] Eval attempt $attempt..."
        cleanup_gpus
        sleep 10

        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        LD_LIBRARY_PATH="/mnt/vast/jiaxin/miniforge3/envs/olmes/lib:${LD_LIBRARY_PATH}" \
        PATH="/mnt/vast/jiaxin/miniforge3/envs/olmes/bin:$PATH" \
        $OLMES_ENV/olmes \
            --model $HF_SFT_DIR \
            --model-type vllm \
            --model-args '{"trust_remote_code": true, "max_length": 8192}' \
            --task \
                minerva_math_500::olmo3:midtrain \
                gpqa::olmo3:adapt \
                gsm8k::olmo3:adapt \
                ifeval::olmo3:adapt \
                ifbench::olmo3:adapt \
                codex_humanevalplus::olmo3:adapt \
                livecodebench_codegeneration::olmo3:adapt \
                aime:2024::olmo3:adapt \
                aime:2025::olmo3:adapt \
                popqa::olmo3:adapt \
            --output-dir $EVAL_DIR \
            > /tmp/eval_step${STEP}.log 2>&1 && break

        echo "[$(date +%H:%M:%S)] Eval attempt $attempt failed, retrying..."
    done

    echo "[$(date +%H:%M:%S)] Evaluation complete for step${STEP}"

    # Extract results
    echo "=== Results for step${STEP} ==="
    grep "'primary_score'" /tmp/eval_step${STEP}.log || echo "No results found"
    echo ""

done

echo ""
echo "================================================================"
echo "=== ALL DONE ==="
echo "================================================================"
echo "Finished at: $(date)"
