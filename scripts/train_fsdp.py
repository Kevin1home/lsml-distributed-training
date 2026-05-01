#!/usr/bin/env python3
"""
FSDP training script.
Covers: Task 2 (sharding strategies), Task 3 (CPU offload), Task 4 (scaling), Bonus A (410m).

Usage examples:

Task 2 — 4 GPU, three sharding strategies:
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \\
        scripts/train_fsdp.py --strategy NO_SHARD --experiment-name p2-no-shard

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \\
        scripts/train_fsdp.py --strategy SHARD_GRAD_OP --experiment-name p2-shard-grad-op

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \\
        scripts/train_fsdp.py --strategy FULL_SHARD --experiment-name p2-full-shard

Task 3 — 2 GPU, CPU offload combinations:
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \\
        scripts/train_fsdp.py --strategy FULL_SHARD --experiment-name p3-full-shard

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \\
        scripts/train_fsdp.py --strategy FULL_SHARD --cpu-offload --experiment-name p3-cpu-offload

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \\
        scripts/train_fsdp.py --strategy FULL_SHARD --cpu-offload --activation-checkpointing \\
        --experiment-name p3-cpu-offload-ac

Task 4 — scaling 1/2/4 GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_fsdp.py \\
        --strategy FULL_SHARD --n-gpus 1 --experiment-name p4-1gpu

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \\
        scripts/train_fsdp.py --strategy FULL_SHARD --experiment-name p4-2gpu

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \\
        scripts/train_fsdp.py --strategy FULL_SHARD --experiment-name p4-4gpu

Bonus A — 410m model:
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \\
        scripts/train_fsdp.py --model EleutherAI/pythia-410m \\
        --strategy FULL_SHARD --experiment-name bonus-410m-full-shard
"""

import argparse
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, default_data_collator

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = 'EleutherAI/pythia-160m'
DATASET_NAME = 'wikitext'
DATASET_CONFIG = 'wikitext-103-v1'

SEED = 42
SEQ_LEN = 512
GLOBAL_BATCH_TOKENS = 131_072
LR = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOG_INTERVAL = 10
EVAL_INTERVAL = 200

# ── Distributed helpers ───────────────────────────────────────────────────────

def setup_dist():
    rank = int(os.getenv('RANK', '0'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size,
                                device_id=device)
    return rank, local_rank, world_size, device


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def rank0_first(rank: int):
    """Run rank 0 first, then all others (useful for downloads)."""
    if rank == 0:
        yield
    if dist.is_initialized():
        dist.barrier()
    if rank > 0:
        yield
    if dist.is_initialized():
        dist.barrier()


def setup_logging(exp_dir: Path, rank: int) -> logging.Logger:
    fmt = f'[rank={rank}] [%(asctime)s] %(levelname)s: %(message)s'
    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(exp_dir / 'train.log'))
    logging.basicConfig(format=fmt, level=logging.INFO, handlers=handlers, force=True)
    return logging.getLogger(__name__)


def gb(bytes_val: int) -> float:
    return bytes_val / (1024 ** 3)


# ── Data ──────────────────────────────────────────────────────────────────────

def tokenize_dataset(dataset, tokenizer, seq_len: int):
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=False, padding=False)

    tokenized = dataset.map(tokenize_fn, batched=True,
                            remove_columns=dataset.column_names, desc='Tokenizing')

    def chunk_fn(examples):
        concat = {k: sum(examples[k], []) for k in examples}
        total = (len(concat['input_ids']) // seq_len) * seq_len
        result = {k: [v[i:i + seq_len] for i in range(0, total, seq_len)]
                  for k, v in concat.items()}
        result['labels'] = result['input_ids'].copy()
        return result

    chunked = tokenized.map(chunk_fn, batched=True, desc='Chunking')
    chunked.set_format(type='torch')
    return chunked


def load_data(tokenizer, seq_len: int, rank: int):
    with rank0_first(rank):
        raw = load_dataset(DATASET_NAME, DATASET_CONFIG)
        train_ds = tokenize_dataset(raw['train'], tokenizer, seq_len)
        val_ds = tokenize_dataset(raw['validation'], tokenizer, seq_len)
    return train_ds, val_ds


def compute_grad_accum(seq_len: int, local_bs: int, dp_size: int) -> int:
    return max(1, GLOBAL_BATCH_TOKENS // (seq_len * local_bs * dp_size))


# ── Model wrapping ────────────────────────────────────────────────────────────

def wrap_model(model, strategy: str, cpu_offload: bool, device, rank: int, world_size: int):
    """
    Wrap model with DDP or FSDP2 depending on strategy.
    NO_SHARD  → DDP
    SHARD_GRAD_OP / FULL_SHARD → FSDP2 (torch.distributed._composable.fsdp)
    """
    if strategy == 'NO_SHARD' or world_size == 1:
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[rank % torch.cuda.device_count()])
        return model

    # FSDP2 API
    from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy
    from torch.distributed._composable.fsdp._fsdp_api import OffloadPolicy

    # Apply per-transformer-block sharding for better memory efficiency
    # Walk through transformer layers and shard each block individually
    _shard_transformer_blocks(model, strategy, cpu_offload, device)

    # Shard the top-level model
    reshard_after_forward = (strategy == 'FULL_SHARD')

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    fully_shard(
        model,
        reshard_after_forward=reshard_after_forward,
        offload_policy=offload_policy,
    )

    return model


def _shard_transformer_blocks(model, strategy: str, cpu_offload: bool, device):
    """Shard individual transformer blocks before sharding the whole model."""
    from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy

    reshard_after_forward = (strategy == 'FULL_SHARD')
    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    # Pythia / GPT-NeoX block structure: model.gpt_neox.layers
    layers = None
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        layers = model.gpt_neox.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers

    if layers is not None:
        for layer in layers:
            fully_shard(layer,
                        reshard_after_forward=reshard_after_forward,
                        offload_policy=offload_policy)


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, rank: int, world_size: int,
             max_batches: int = 50) -> float:
    model.eval()
    total_loss = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss
        count += 1

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    model.train()
    avg_loss = (total_loss / count).item()
    return math.exp(avg_loss)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='FSDP distributed training')
    p.add_argument('--model', default=MODEL_NAME)
    p.add_argument('--strategy', choices=['NO_SHARD', 'SHARD_GRAD_OP', 'FULL_SHARD'],
                   default='FULL_SHARD')
    p.add_argument('--dtype', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--cpu-offload', action='store_true')
    p.add_argument('--activation-checkpointing', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-epochs', type=int, default=1)
    p.add_argument('--n-gpus', type=int, default=None,
                   help='Informational: number of GPUs (auto-detected from world_size)')
    p.add_argument('--experiment-name', required=True)
    p.add_argument('--log-dir', default='logs/')
    p.add_argument('--no-eval', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)

    rank, local_rank, world_size, device = setup_dist()

    # Create experiment directory on rank 0
    exp_dir = Path(args.log_dir) / args.experiment_name
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        exp_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    logger = setup_logging(exp_dir, rank)

    if rank == 0:
        logger.info(f'Experiment: {args.experiment_name}')
        logger.info(f'strategy={args.strategy}  dtype={args.dtype}  '
                    f'cpu_offload={args.cpu_offload}  '
                    f'activation_checkpointing={args.activation_checkpointing}  '
                    f'world_size={world_size}')

    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32

    # ── Model & tokenizer ──────────────────────────────────────────────────
    with rank0_first(rank):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model, use_cache=False)
    with device:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
        if rank == 0:
            logger.info('Activation checkpointing enabled')

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f'Parameters: {n_params / 1e6:.1f}M')

    # ── Wrap with DDP / FSDP ──────────────────────────────────────────────
    model = wrap_model(args.strategy, args.cpu_offload, device, rank, world_size) \
        if False else wrap_model(model, args.strategy, args.cpu_offload, device, rank, world_size)

    # ── Data ──────────────────────────────────────────────────────────────
    if rank == 0:
        logger.info('Loading dataset...')
    train_ds, val_ds = load_data(tokenizer, SEQ_LEN, rank)

    grad_accum = compute_grad_accum(SEQ_LEN, args.batch_size, world_size)
    if rank == 0:
        logger.info(f'batch_size={args.batch_size}  grad_accum={grad_accum}  '
                    f'world_size={world_size}  '
                    f'=> {SEQ_LEN * args.batch_size * grad_accum * world_size} tokens/step')

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                       shuffle=True, drop_last=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False) if world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              drop_last=(train_sampler is None),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler,
                            num_workers=4, pin_memory=True)

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
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
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
            tokens_seen += batch['input_ids'].numel() * world_size

            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                elapsed = time.time() - step_start
                step_tokens = SEQ_LEN * args.batch_size * grad_accum * world_size
                tps = step_tokens / elapsed
                peak_mem = gb(torch.cuda.max_memory_allocated(device))

                if rank == 0 and global_step % LOG_INTERVAL == 0:
                    logger.info(
                        f'step={global_step:5d}  loss={accum_loss:.4f}  '
                        f'tps={tps:.0f}  peak_mem={peak_mem:.3f}GB'
                    )

                if rank == 0:
                    log_rows.append({
                        'step': global_step,
                        'train_loss': accum_loss,
                        'tps': tps,
                        'peak_mem_gb': peak_mem,
                    })

                accum_loss = 0.0
                step_start = time.time()

                if not args.no_eval and global_step % EVAL_INTERVAL == 0:
                    val_ppl = evaluate(model, val_loader, device, rank, world_size)
                    if rank == 0:
                        logger.info(f'  >>> val_ppl={val_ppl:.2f}')
                        log_rows[-1]['val_ppl'] = val_ppl

    # ── Final eval ────────────────────────────────────────────────────────
    if rank == 0:
        logger.info('Final evaluation...')
    val_ppl = evaluate(model, val_loader, device, rank, world_size)
    peak_mem = gb(torch.cuda.max_memory_allocated(device))
    total_time = time.time() - train_start
    total_tps = tokens_seen / total_time

    if rank == 0:
        logger.info(f'FINAL: val_ppl={val_ppl:.4f}  peak_mem={peak_mem:.3f}GB  '
                    f'throughput={total_tps:.0f} tok/s  time={total_time:.0f}s')

        results = {
            'experiment': args.experiment_name,
            'model': args.model,
            'strategy': args.strategy,
            'dtype': args.dtype,
            'cpu_offload': args.cpu_offload,
            'activation_checkpointing': args.activation_checkpointing,
            'world_size': world_size,
            'val_ppl': val_ppl,
            'peak_mem_gpu_gb': peak_mem,
            'throughput_tps': total_tps,
            'throughput_per_gpu_tps': total_tps / world_size,
            'total_time_s': total_time,
            'total_steps': global_step,
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

    cleanup_dist()


if __name__ == '__main__':
    main()
