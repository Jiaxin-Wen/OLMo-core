"""
OLMo3-7B pre-training script.

Uses a single .npy data file in exact sequential order (no shuffle).
Global batch size ~4M tokens, achieved via gradient accumulation.

Usage:
    torchrun --nproc-per-node=8 pretrain_olmo3_7B.py \
        --save-folder /path/to/checkpoints \
        --data-path /path/to/data.npy

    # Dry run (print config and exit):
    python pretrain_olmo3_7B.py --save-folder /tmp/test --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import rich

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyDatasetDType,
    TokenizerConfig,
)
from olmo_core.data.collator import DataCollator
from olmo_core.data.data_loader import NumpyDataLoaderBase
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size, get_fs_local_rank
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
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

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # ~4M tokens
LR = 3e-4
WARMUP_STEPS = 400
INIT_SEED = 12536


def build_config(opts: argparse.Namespace):
    tokenizer_config = TokenizerConfig.dolma2()

    # ── Model ────────────────────────────────────────────────────────────
    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_config = NumpyFSLDatasetConfig(
        paths=[opts.data_path],
        tokenizer=tokenizer_config,
        sequence_length=SEQUENCE_LENGTH,
        dtype=NumpyDatasetDType.uint32,
        work_dir=opts.work_dir,
    )

    # ── Data loader (config only used for dry-run display; real loader
    #    is built manually with shuffle=False) ────────────────────────────
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    # ── Train module ─────────────────────────────────────────────────────
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=8 * SEQUENCE_LENGTH,
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
        scheduler=CosWithWarmup(warmup_steps=WARMUP_STEPS),
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
    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=50,
            cancel_check_interval=10,
            max_duration=Duration.epochs(1),
            work_dir=opts.work_dir,
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name or "olmo3-7b-pretrain",
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
    parser.add_argument("--work-dir", type=str, default=None)
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
        shuffle=False,   # <-- preserve exact data order
        num_workers=4,
        target_device_type=get_default_device().type,
    )

    # ── Build trainer ────────────────────────────────────────────────────
    trainer = cfg["trainer"].build(train_module, data_loader)

    for callback in trainer.callbacks.values():
        if isinstance(callback, ConfigSaverCallback):
            callback.config = {k: str(v) for k, v in cfg.items()}
            break

    # ── Load checkpoint if resuming ──────────────────────────────────────
    trainer.maybe_load_checkpoint()

    # ── Train ────────────────────────────────────────────────────────────
    trainer.fit()

    teardown_training_environment()


if __name__ == "__main__":
    main()
