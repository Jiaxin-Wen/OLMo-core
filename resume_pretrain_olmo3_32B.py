"""
OLMo3-32B resume pre-training script.

Loads from an official OLMo3 32B stage1 checkpoint (with optimizer state)
and continues pre-training on custom data. Uses the exact same LR schedule
as the original 5.9T token stage1 pre-training.

Usage:
    torchrun --nproc-per-node=8 --nnodes=2 --master_addr=... --master_port=... --node_rank=... \
        resume_pretrain_olmo3_32B.py \
        --save-folder /path/to/checkpoints \
        --data-path /path/to/data.npy \
        --load-path https://olmo-checkpoints.org/ai2-llm/stego32-highlr-filter3/step233000/ \
        --max-steps 3000
"""

import argparse
import logging
import sys
from pathlib import Path

import rich

from olmo_core.config import DType
from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyDatasetDType,
    TokenizerConfig,
)
from olmo_core.data.collator import DataCollator
from olmo_core.data.data_loader import NumpyDataLoaderBase
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size, get_fs_local_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import get_default_device, seed_all

log = logging.getLogger(__name__)

# ── Hyperparameters matching original OLMo3 32B stage1 ───────────────────────
SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024  # 8M tokens, matching original OLMo3 32B
LR = 6e-4
WARMUP_STEPS = 2000
ALPHA_F = 0.1
# t_max for the original 5.93T cosine schedule (truncated at 5.5T):
# 5.93e12 / 8388608 ≈ 706,911
# This ensures the LR schedule matches the original training exactly
ORIGINAL_T_MAX = 706911
INIT_SEED = 12536


def build_config(opts: argparse.Namespace):
    tokenizer_config = TokenizerConfig.dolma2()

    # ── Model ────────────────────────────────────────────────────────────
    model_config = TransformerConfig.olmo3_32B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName(opts.attn_backend),
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_config = NumpyFSLDatasetConfig(
        paths=[opts.data_path],
        tokenizer=tokenizer_config,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=SEQUENCE_LENGTH,  # match original: max(8192, seq_len)
        dtype=NumpyDatasetDType.uint32,
        work_dir=opts.work_dir,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    # ── Data loader config (for display only) ────────────────────────────
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        ignore_fingerprint_mismatch=True,
    )

    # ── Train module ─────────────────────────────────────────────────────
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # 1 sequence per microbatch (match official)
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts=dict(weight_decay=0.0),
                )
            ],
        ),
        # Use t_max to match the original 5.9T token LR schedule exactly
        scheduler=CosWithWarmup(
            warmup=WARMUP_STEPS,
            alpha_f=ALPHA_F,
            t_max=ORIGINAL_T_MAX,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    # We load model+optimizer state but NOT trainer state, because the
    # original checkpoint's data loader state (batches_processed) would
    # cause our smaller dataset to seek to the wrong position.
    # Instead, we manually set global_step after loading.
    target_step = opts.resume_step + opts.max_steps

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            load_path=opts.load_path,
            load_strategy=LoadStrategy.always,
            load_optim_state=True,
            load_trainer_state=False,
            metrics_collect_interval=50,
            cancel_check_interval=10,
            max_duration=Duration.steps(target_step),
            work_dir=opts.work_dir,
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=200,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name or f"olmo3-32b-resume-from-{opts.resume_step}",
                cancel_check_interval=10,
                enabled=True,
            ),
        )
    )

    return dict(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        tokenizer=tokenizer_config,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--save-folder", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--load-path", type=str, required=True,
                        help="URL or path to the checkpoint to resume from")
    parser.add_argument("--resume-step", type=int, required=True,
                        help="The global step of the checkpoint being loaded (e.g. 233000)")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Number of additional steps to train")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--attn-backend", type=str, default="flash_2",
                        choices=["flash_2", "flash_3", "torch", "te"],
                        help="Attention backend. Default flash_2 matches official training.")
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()

    if opts.work_dir is None:
        opts.work_dir = opts.save_folder if not opts.save_folder.startswith(("s3://", "gs://")) \
            else "/tmp/olmo-core/dataset-cache"

    cfg = build_config(opts)

    if opts.dry_run:
        from olmo_core.train import prepare_cli_environment
        prepare_cli_environment()
        for k, v in cfg.items():
            rich.print(f"\n[bold]{k}[/bold]:")
            rich.print(v)

        # Show expected LR at resume point
        import math
        eta_min = LR * ALPHA_F
        c = opts.resume_step - WARMUP_STEPS
        t = ORIGINAL_T_MAX - WARMUP_STEPS
        lr_at_resume = eta_min + (LR - eta_min) * (1 + math.cos(math.pi * c / t)) / 2
        rich.print(f"\n[bold]LR at step {opts.resume_step}[/bold]: {lr_at_resume:.6f}")
        rich.print(f"[bold]Target step[/bold]: {opts.resume_step + opts.max_steps}")
        return

    # ── Set up distributed ───────────────────────────────────────────────
    prepare_training_environment()
    seed_all(INIT_SEED)

    # ── Build model & train module ───────────────────────────────────────
    model = cfg["model"].build(init_device="meta")
    train_module = cfg["train_module"].build(model)

    # ── Build dataset ────────────────────────────────────────────────────
    dataset = cfg["dataset"].build()
    if cfg["dataset"].work_dir is not None:
        dataset.work_dir = Path(cfg["dataset"].work_dir)
    dataset.prepare()

    # ── Build data loader with shuffle=False ─────────────────────────────
    dp_process_group = train_module.dp_process_group
    data_loader = NumpyDataLoaderBase.wrap_numpy_dataset(
        dataset,
        global_batch_size=GLOBAL_BATCH_SIZE,
        collator=DataCollator(pad_token_id=cfg["tokenizer"].pad_token_id),
        work_dir=cfg["dataset"].work_dir or dataset.work_dir,
        dp_world_size=get_world_size(dp_process_group),
        dp_rank=get_rank(dp_process_group),
        fs_local_rank=get_fs_local_rank(),
        seed=34521,
        shuffle=False,
        num_workers=4,
        target_device_type=get_default_device().type,
        ignore_fingerprint_mismatch=True,
    )

    # ── Build trainer ────────────────────────────────────────────────────
    trainer = cfg["trainer"].build(train_module, data_loader)

    for callback in trainer.callbacks.values():
        if isinstance(callback, ConfigSaverCallback):
            callback.config = {k: str(v) for k, v in cfg.items()}
            break

    # ── Load checkpoint (model + optimizer, no trainer state) ────────────
    log.info(f"Loading checkpoint from {cfg['trainer'].load_path}...")
    trainer.load_checkpoint(cfg["trainer"].load_path, load_trainer_state=False)

    # Manually set global_step so the LR schedule resumes at the right point.
    # We don't load trainer state to avoid the data loader seeking to the
    # original checkpoint's batches_processed position in our smaller dataset.
    trainer.global_step = opts.resume_step
    trainer.global_train_tokens_seen = opts.resume_step * GLOBAL_BATCH_SIZE
    log.info(f"Set global_step={trainer.global_step}, tokens_seen={trainer.global_train_tokens_seen:,}")

    # ── Train ────────────────────────────────────────────────────────────
    trainer.fit()

    teardown_training_environment()


if __name__ == "__main__":
    main()
