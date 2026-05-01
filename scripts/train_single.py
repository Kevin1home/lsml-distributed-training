#!/usr/bin/env python3
"""
Single-GPU baseline training script.
Covers: Task 1 (fp32, bf16, bf16 + activation checkpointing)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_single.py \
        --dtype bf16 \
        --experiment-name p1-bf16 \
        --log-dir logs/

    CUDA_VISIBLE_DEVICES=0 python scripts/train_single.py \
        --dtype fp32 \
        --experiment-name p1-fp32 \
        --log-dir logs/

    CUDA_VISIBLE_DEVICES=0 python scripts/train_single.py \
        --dtype bf16 --activation-checkpointing \
        --experiment-name p1-bf16-ac \
        --log-dir logs/
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, default_data_collator

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = 'EleutherAI/pythia-160m'
DATASET_NAME = 'wikitext'
DATASET_CONFIG = 'wikitext-103-v1'

SEED = 42
SEQ_LEN = 512
GLOBAL_BATCH_TOKENS = 131_072   # fixed token budget per step
LR = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOG_INTERVAL = 10
EVAL_INTERVAL = 200

# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_logging(exp_dir: Path, rank: int = 0) -> logging.Logger:
    log_file = exp_dir / 'train.log'
    fmt = '[%(asctime)s] %(levelname)s: %(message)s'
    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(format=fmt, level=logging.INFO, handlers=handlers, force=True)
    return logging.getLogger(__name__)


def tokenize_dataset(dataset, tokenizer, seq_len: int):
    """Tokenize and chunk the dataset into fixed-length sequences."""
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=False, padding=False)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc='Tokenizing',
    )

    # Concatenate and chunk into seq_len blocks
    def chunk_fn(examples):
        concat = {k: sum(examples[k], []) for k in examples}
        total = len(concat['input_ids'])
        # drop last incomplete chunk
        total = (total // seq_len) * seq_len
        result = {k: [v[i:i + seq_len] for i in range(0, total, seq_len)] for k, v in concat.items()}
        result['labels'] = result['input_ids'].copy()
        return result

    chunked = tokenized.map(chunk_fn, batched=True, desc='Chunking')
    chunked.set_format(type='torch')
    return chunked


def load_data(tokenizer, seq_len: int):
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)
    train_ds = tokenize_dataset(raw['train'], tokenizer, seq_len)
    val_ds = tokenize_dataset(raw['validation'], tokenizer, seq_len)
    return train_ds, val_ds


def compute_grad_accum(seq_len: int, local_bs: int, dp_size: int = 1) -> int:
    """Compute gradient accumulation steps to hit GLOBAL_BATCH_TOKENS."""
    return GLOBAL_BATCH_TOKENS // (seq_len * local_bs * dp_size)


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        count += 1
    model.train()
    avg_loss = total_loss / max(count, 1)
    return math.exp(avg_loss)   # perplexity


def gb(bytes_val: int) -> float:
    return bytes_val / (1024 ** 3)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Single-GPU baseline training')
    p.add_argument('--model', default=MODEL_NAME)
    p.add_argument('--dtype', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--activation-checkpointing', action='store_true')
    p.add_argument('--batch-size', type=int, default=8,
                   help='Local batch size (sequences per GPU)')
    p.add_argument('--num-epochs', type=int, default=1)
    p.add_argument('--experiment-name', required=True)
    p.add_argument('--log-dir', default='logs/')
    p.add_argument('--no-eval', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(SEED)

    # Directories
    exp_dir = Path(args.log_dir) / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(exp_dir)
    logger.info(f'Experiment: {args.experiment_name}')
    logger.info(f'dtype={args.dtype}  activation_checkpointing={args.activation_checkpointing}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32

    # ── Model & tokenizer ──────────────────────────────────────────────────
    logger.info(f'Loading model {args.model}')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model, use_cache=False)
    with device:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info('Activation checkpointing enabled')

    logger.info(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info('Loading dataset...')
    train_ds, val_ds = load_data(tokenizer, SEQ_LEN)

    grad_accum = compute_grad_accum(SEQ_LEN, args.batch_size)
    logger.info(f'batch_size={args.batch_size}  grad_accum={grad_accum}  '
                f'=> {SEQ_LEN * args.batch_size * grad_accum} tokens/step')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=4, pin_memory=True)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── Training ──────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    global_step = 0
    tokens_seen = 0
    train_start = time.time()
    step_start = time.time()
    log_rows = []

    for epoch in range(args.num_epochs):
        logger.info(f'=== Epoch {epoch + 1}/{args.num_epochs} ===')

        accum_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            is_last_accum = (batch_idx + 1) % grad_accum == 0

            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            tokens_seen += batch['input_ids'].numel()

            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                elapsed = time.time() - step_start
                step_tokens = SEQ_LEN * args.batch_size * grad_accum
                tps = step_tokens / elapsed
                peak_mem = gb(torch.cuda.max_memory_allocated(device))

                if global_step % LOG_INTERVAL == 0:
                    logger.info(
                        f'step={global_step:5d}  loss={accum_loss:.4f}  '
                        f'tps={tps:.0f}  peak_mem={peak_mem:.3f}GB'
                    )

                log_rows.append({
                    'step': global_step,
                    'train_loss': accum_loss,
                    'tps': tps,
                    'peak_mem_gb': peak_mem,
                })

                accum_loss = 0.0
                step_start = time.time()

                # Validation
                if not args.no_eval and global_step % EVAL_INTERVAL == 0:
                    val_ppl = evaluate(model, val_loader, device)
                    logger.info(f'  >>> val_ppl={val_ppl:.2f}')
                    log_rows[-1]['val_ppl'] = val_ppl

    # ── Final eval ────────────────────────────────────────────────────────
    logger.info('Final evaluation...')
    val_ppl = evaluate(model, val_loader, device)
    peak_mem = gb(torch.cuda.max_memory_allocated(device))
    total_time = time.time() - train_start
    total_tps = tokens_seen / total_time

    logger.info(f'FINAL: val_ppl={val_ppl:.4f}  peak_mem={peak_mem:.3f}GB  '
                f'throughput={total_tps:.0f} tok/s  time={total_time:.0f}s')

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        'experiment': args.experiment_name,
        'dtype': args.dtype,
        'activation_checkpointing': args.activation_checkpointing,
        'val_ppl': val_ppl,
        'peak_mem_gb': peak_mem,
        'throughput_tps': total_tps,
        'total_time_s': total_time,
        'total_steps': global_step,
        'tokens_seen': tokens_seen,
    }
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    import csv
    with open(exp_dir / 'training_log.csv', 'w', newline='') as f:
        if log_rows:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

    logger.info(f'Results saved to {exp_dir}')


if __name__ == '__main__':
    main()
