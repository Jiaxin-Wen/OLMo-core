# OLMo-core SFT Training & Evaluation

This fork of [allenai/OLMo-core](https://github.com/allenai/OLMo-core) contains scripts and fixes for running OLMo3 32B SFT on non-SLURM H100 clusters and evaluating with [olmes](https://github.com/allenai/olmes).

## Key Changes from Upstream

### 1. Data Processing Bug Fix (Related: [allenai/OLMo-core#617](https://github.com/allenai/OLMo-core/issues/617))

**Problem**: OLMo-core's `load_array_slice()` in `src/olmo_core/data/utils.py` reads raw bytes from data files using `get_bytes_range()` with zero byte offset. However, files saved with `np.save()` include a 128-byte `.npy` header. The reader assumes headerless raw binary, so it interprets the header bytes as token IDs — producing garbage values (e.g., `1297436307`) that exceed the vocab size, triggering a CUDA device-side assert at runtime.

**Symptom**: Training crashes with `CUDA error: device-side assert triggered` at a deterministic step (in our case, always step 25), always on the same rank. The crash appears to be data-dependent but is actually caused by byte-shifted reads.

**Fix**: Strip the `.npy` header from tokenized data files before training:

```python
import numpy as np

for name in ['token_ids_part_0', 'labels_mask_0']:
    arr = np.load(f'{name}.npy')
    with open(f'{name}.npy', 'wb') as f:
        f.write(arr.tobytes())  # Write raw binary without numpy header
```

**Verification**: You can detect this issue by comparing `np.load()` vs `np.memmap()`:
```python
tokens_load = np.load('token_ids_part_0.npy')       # Correctly skips header
tokens_mmap = np.memmap('token_ids_part_0.npy', dtype=np.uint32, mode='r')  # Reads from byte 0
# If file has header: tokens_mmap will have 32 extra garbage values at the start
```

### 2. vLLM 0.11 API Patch for olmes/lm_eval

**Problem**: olmes depends on `lm_eval` (EleutherAI's lm-evaluation-harness), which calls `LLM.generate(prompt_token_ids=requests, ...)`. In vLLM 0.11, the `prompt_token_ids` parameter was removed in favor of the `TokensPrompt` input type.

**Symptom**: `TypeError: LLM.generate() got an unexpected keyword argument 'prompt_token_ids'`

**Fix**: Patch `lm_eval/models/vllm_causallms.py` (in the olmes conda env):

```python
# Before (broken with vLLM 0.11):
outputs = self.model.generate(
    prompt_token_ids=requests,
    sampling_params=sampling_params,
)

# After (compatible with vLLM 0.11):
from vllm.inputs import TokensPrompt
prompts = [TokensPrompt(prompt_token_ids=req) for req in requests]
outputs = self.model.generate(
    prompts,
    sampling_params=sampling_params,
)
```

Apply this change in two locations in the file (the main generate path and the ray remote path).

**Note**: This patch may not be needed if you use a compatible vLLM + lm_eval version pair.

### 3. Checkpoint Save Without Optimizer State

Added `skip_optim_save` field to `CheckpointerConfig` and `Checkpointer` in `src/olmo_core/train/checkpoint.py` to allow saving model weights only (no optimizer state) during SFT, significantly reducing checkpoint size and save time.

### 4. Removed `ephemeral_save_interval` Default

Removed the default `ephemeral_save_interval=500` from `src/scripts/train/sft/Olmo-3-32B-SFT.py` to avoid conflicts when overriding `save_interval` to a smaller value.

## New Scripts

### Training

- **`src/scripts/train/sft/launch-2node-32B-sft.sh`**: Multi-node launch script for 2x H100 nodes (no SLURM). Configures FSDP across 16 GPUs, InfiniBand for NCCL, full activation checkpointing, and per-epoch checkpointing.

- **`run_sft_pipeline.sh`**: Sequential pipeline that downloads, converts, trains, and evaluates multiple OLMo3 stage1 checkpoints.

### Testing

- **`test_nccl_ib.py`**: Quick NCCL InfiniBand bandwidth test across 2 nodes.

## Environment Setup

### Training (venv)
```bash
python -m venv .venv
.venv/bin/pip install torch
.venv/bin/pip install flash-attn --no-build-isolation
.venv/bin/pip install -e '.[all]'
```

### Evaluation (conda, requires Python < 3.13)
```bash
conda create -n olmes python=3.12 -y
conda activate olmes
pip install -e "/path/to/olmes[gpu]"
# Then apply the vLLM patch above if needed
```

## Multi-Node Training Notes

- Uses InfiniBand for NCCL (not Ethernet): set `NCCL_IB_DISABLE=0`, `GLOO_SOCKET_IFNAME=<ethernet_iface>`
- FSDP shard across all GPUs (not HSDP within node) for H100 80GB memory constraints
- Full activation checkpointing required for 32B model on H100
- Always kill zombie GPU processes on all nodes before launching new jobs

## Evaluation with olmes

We use the `olmo3:adapt` evaluation config from olmes:
- Temperature 0.6, top_p 0.95, sampling enabled
- 0-shot for all benchmarks
- `max_length=8192` (model context window)
- Multi-sample with pass@k for math/coding, single sample for IF/knowledge

See the olmes repo for the full `olmo3:adapt` task suite definition.
