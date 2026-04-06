#!/bin/bash
# Wait for 500K training to finish, kill pipeline, then eval 574K, 573K, 500K
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

OLMES_ENV=/mnt/vast/jiaxin/miniforge3/envs/olmes/bin
CKPT_BASE=/mnt/vast/jiaxin/checkpoints
WORKER_ADDR=91.239.86.229

cleanup_gpus() {
    pkill -9 -f "Olmo-3-32B-SFT\|oe_eval\|olmes\|vllm" 2>/dev/null || true
    ssh $WORKER_ADDR "pkill -9 -f 'Olmo-3-32B-SFT\|oe_eval\|olmes\|vllm'" 2>/dev/null || true
    sleep 5
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 $pid 2>/dev/null || true
    done
    sleep 5
}

# Step 1: Wait for 500K training to complete
echo "[$(date +%H:%M:%S)] Waiting for step 500K training to finish..."
while true; do
    if grep -q "Training complete" /tmp/sft_node0_step500000.log 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] 500K training complete!"
        break
    fi
    sleep 30
done

# Step 2: Wait for HF conversion
echo "[$(date +%H:%M:%S)] Waiting for HF conversion..."
while true; do
    HF_DIR=$CKPT_BASE/ubuntu/olmo-sft/olmo3-32b-sft-instruct-100k-step500000/step246-hf
    if [ -d "$HF_DIR" ] && [ "$(ls $HF_DIR/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] HF conversion complete!"
        break
    fi
    sleep 10
done

# Step 3: Kill the pipeline (before it starts 493K eval/training)
echo "[$(date +%H:%M:%S)] Killing pipeline..."
pkill -f "run_sft_pipeline" 2>/dev/null || true
sleep 5
cleanup_gpus

# Step 4: Run evals for 574K, 573K, 500K
for STEP in 574000 573000 500000; do
    HF_SFT_DIR=$CKPT_BASE/ubuntu/olmo-sft/olmo3-32b-sft-instruct-100k-step${STEP}/step246-hf
    EVAL_DIR=/mnt/vast/jiaxin/eval-results-step${STEP}
    mkdir -p $EVAL_DIR

    echo ""
    echo "[$(date +%H:%M:%S)] ========== Evaluating step ${STEP} =========="
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
            codex_humanevalplus::olmo3:adapt \
            livecodebench_codegeneration::olmo3:adapt \
            aime:2024::olmo3:adapt \
            aime:2025::olmo3:adapt \
            popqa::olmo3:adapt \
        --output-dir $EVAL_DIR \
        > /tmp/eval_step${STEP}.log 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Eval step${STEP} complete!"
        grep "'primary_score'" /tmp/eval_step${STEP}.log || true
    else
        echo "[$(date +%H:%M:%S)] Eval step${STEP} FAILED"
    fi
done

echo ""
echo "[$(date +%H:%M:%S)] All three evals done. Resume pipeline manually."
