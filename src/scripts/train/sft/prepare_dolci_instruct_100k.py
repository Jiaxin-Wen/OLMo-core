"""
Prepare a 100K subset of Dolci-Instruct-SFT for OLMo-core SFT training.

Strategy: downsample first (stratified by domain), then tokenize only the
selected 100K. This avoids tokenizing the full 2.15M dataset.

Usage:
    export HF_HOME=/workspace-vast/pretrained_ckpts

    python src/scripts/train/sft/prepare_dolci_instruct_100k.py \
        --output_dir /workspace-vast/wenj/data/dolci-instruct-100k \
        --max_seq_length 8192 \
        --num_samples 100000
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257


def tokenize_conversation(tokenizer, messages):
    """
    Tokenize a conversation using the dolma2 chatml format and produce token_ids
    and a label_mask. Does NOT truncate -- caller handles truncation.

    The label_mask is True for assistant response tokens (loss targets)
    and False for everything else (user/system prompts, special tokens).
    """
    token_ids = []
    label_mask = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")
        if content is None:
            continue

        # Format: <|im_start|>{role}\n{content}<|im_end|>\n
        header_ids = tokenizer.encode(
            f"<|im_start|>{role}\n", add_special_tokens=False
        )
        content_ids = tokenizer.encode(
            f"{content}<|im_end|>\n", add_special_tokens=False
        )

        token_ids.extend(header_ids)
        label_mask.extend([False] * len(header_ids))

        token_ids.extend(content_ids)
        if role == "assistant":
            label_mask.extend([True] * len(content_ids))
        else:
            label_mask.extend([False] * len(content_ids))

    return token_ids, label_mask


def stratified_sample_indices(dataset, num_samples, rng):
    """
    Sample indices using stratified sampling by domain to preserve the original
    dataset's domain proportions (following the paper's ablation practice).
    """
    domain_indices = defaultdict(list)
    for i, domain in enumerate(dataset["domain"]):
        domain_indices[domain].append(i)

    total = len(dataset)
    selected = []

    # Compute per-domain quotas proportional to original distribution
    domain_quotas = {}
    allocated = 0
    domains = sorted(domain_indices.keys())
    for i, domain in enumerate(domains):
        if i == len(domains) - 1:
            domain_quotas[domain] = num_samples - allocated
        else:
            quota = int(round(num_samples * len(domain_indices[domain]) / total))
            domain_quotas[domain] = quota
            allocated += quota

    log.info("Domain sampling quotas:")
    for domain in domains:
        n_available = len(domain_indices[domain])
        quota = domain_quotas[domain]
        log.info(f"  {domain}: {quota} / {n_available} ({100 * quota / num_samples:.1f}%)")

    # Random sample within each domain
    for domain in domains:
        indices = np.array(domain_indices[domain])
        quota = min(domain_quotas[domain], len(indices))
        if quota <= 0:
            continue
        chosen = rng.choice(indices, size=quota, replace=False)
        selected.extend(chosen.tolist())

    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Prepare 100K Dolci-Instruct-SFT subset")
    parser.add_argument(
        "--output_dir", type=str,
        default="/workspace-vast/wenj/data/dolci-instruct-100k",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=8192,
        help="Max sequence length for truncation.",
    )
    parser.add_argument("--num_samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset_name", type=str, default="allenai/Dolci-Instruct-SFT",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    log.info(f"Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)

    log.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    log.info(f"Total examples: {len(dataset)}")

    # Show original domain distribution
    log.info("Original domain distribution:")
    for domain, count in sorted(Counter(dataset["domain"]).items(), key=lambda x: -x[1]):
        log.info(f"  {domain}: {count} ({100 * count / len(dataset):.1f}%)")

    # Step 1: Downsample first (stratified by domain)
    log.info(f"Stratified sampling {args.num_samples} examples...")
    selected_indices = stratified_sample_indices(dataset, args.num_samples, rng)
    log.info(f"Selected {len(selected_indices)} examples")

    subset = dataset.select(selected_indices)

    # Log final domain distribution
    final_dist = Counter(subset["domain"])
    log.info("Selected domain distribution:")
    for domain, count in sorted(final_dist.items(), key=lambda x: -x[1]):
        log.info(f"  {domain}: {count} ({100 * count / len(selected_indices):.1f}%)")

    # Step 2: Tokenize only the selected 100K
    log.info(f"Tokenizing {len(subset)} selected examples...")
    concat_token_ids = []
    concat_label_mask = []
    n_truncated = 0

    for i in range(len(subset)):
        if i % 10_000 == 0:
            log.info(f"  {i}/{len(subset)}...")
        example = subset[i]
        token_ids, label_mask = tokenize_conversation(tokenizer, example["messages"])

        if len(token_ids) > args.max_seq_length:
            n_truncated += 1
            token_ids = token_ids[: args.max_seq_length]
            label_mask = label_mask[: args.max_seq_length]

        concat_token_ids.extend(token_ids)
        concat_label_mask.extend(label_mask)
        # Ensure EOS at document boundary for packing
        if not token_ids or token_ids[-1] != EOS_TOKEN_ID:
            concat_token_ids.append(EOS_TOKEN_ID)
            concat_label_mask.append(False)

    log.info(f"Truncated {n_truncated}/{len(subset)} examples "
             f"({100 * n_truncated / len(subset):.1f}%)")

    concat_token_ids = np.array(concat_token_ids, dtype=np.uint32)
    concat_label_mask = np.array(concat_label_mask, dtype=np.bool_)

    log.info(f"Total tokens: {len(concat_token_ids):,}")
    log.info(
        f"Labeled tokens: {concat_label_mask.sum():,} "
        f"({100 * concat_label_mask.sum() / len(concat_label_mask):.1f}%)"
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "token_ids_part_0.npy", concat_token_ids)
    np.save(output_dir / "labels_mask_0.npy", concat_label_mask)

    meta = {
        "dataset": args.dataset_name,
        "num_samples": len(selected_indices),
        "max_seq_length": args.max_seq_length,
        "total_tokens": int(len(concat_token_ids)),
        "labeled_tokens": int(concat_label_mask.sum()),
        "truncated_examples": n_truncated,
        "seed": args.seed,
        "tokenizer": TOKENIZER_ID,
        "sampling": "stratified by domain, proportional to original distribution",
        "domain_distribution": dict(final_dist),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Saved to {output_dir}")
    log.info("Done!")


if __name__ == "__main__":
    main()
