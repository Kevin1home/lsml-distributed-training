#!/usr/bin/env bash
# run_all.sh — runs all experiments for HW2
# Edit CUDA_VISIBLE_DEVICES to match server's GPU indices.
# Set DRY_RUN=1 to print commands without executing.

set -euo pipefail

export HF_HOME=/data/shared_ml/huggingface
export TRANSFORMERS_CACHE=/data/shared_ml/huggingface
export HF_DATASETS_CACHE=~/hf_datasets_cache

DRY_RUN=${DRY_RUN:-0}
LOG_DIR=${LOG_DIR:-logs}

run() {
    echo ""
    echo "▶ $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        eval "$@"
    fi
}

mkdir -p "$LOG_DIR" plots

export TORCHELASTIC_ERROR_FILE=error.json
export OMP_NUM_THREADS=1

# ─────────────────────────────────────────────────────
# TASK 1 — Single-GPU Baseline
# ─────────────────────────────────────────────────────
echo "========== TASK 1: Single-GPU Baseline =========="

#run "CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_single.py \
#    --dtype fp32 --batch-size 8 \
#    --experiment-name p1-fp32 --log-dir $LOG_DIR"

#run "CUDA_VISIBLE_DEVICES=4 uv run python scripts/train_single.py \
#    --dtype bf16 --batch-size 8 \
#    --experiment-name p1-bf16 --log-dir $LOG_DIR"

#run "CUDA_VISIBLE_DEVICES=4 uv run python scripts/train_single.py \
#    --dtype bf16 --activation-checkpointing --batch-size 8 \
#    --experiment-name p1-bf16-ac --log-dir $LOG_DIR"

# ─────────────────────────────────────────────────────
# TASK 2 — FSDP Strategies (4 GPU)
# ─────────────────────────────────────────────────────
echo "========== TASK 2: FSDP Strategies (4 GPU) =========="

#run "CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun \
#    --nproc-per-node 4 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --strategy NO_SHARD --batch-size 4 \
#    --experiment-name p2-no-shard --log-dir $LOG_DIR"

#run "CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun \
#    --nproc-per-node 4 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --strategy SHARD_GRAD_OP --batch-size 4 \
#    --experiment-name p2-shard-grad-op --log-dir $LOG_DIR"

#run "CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun \
#    --nproc-per-node 4 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --strategy FULL_SHARD --batch-size 4 \
#    --experiment-name p2-full-shard --log-dir $LOG_DIR"

# ─────────────────────────────────────────────────────
# TASK 3 — CPU Offload (2 GPU)
# ─────────────────────────────────────────────────────
echo "========== TASK 3: CPU Offload (2 GPU) =========="

#run "CUDA_VISIBLE_DEVICES=6,7 uv run torchrun \
#    --nproc-per-node 2 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --strategy FULL_SHARD --batch-size 4 \
#    --experiment-name p3-full-shard --log-dir $LOG_DIR"

#run "CUDA_VISIBLE_DEVICES=6,7 uv run torchrun \
#    --nproc-per-node 2 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --strategy FULL_SHARD --cpu-offload --batch-size 4 \
#    --experiment-name p3-cpu-offload --log-dir $LOG_DIR"

#run "CUDA_VISIBLE_DEVICES=6,7 uv run torchrun \
#    --nproc-per-node 2 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --strategy FULL_SHARD --cpu-offload \
#    --activation-checkpointing --batch-size 4 \
#    --experiment-name p3-cpu-offload-ac --log-dir $LOG_DIR"

# ─────────────────────────────────────────────────────
# TASK 4 — Scaling (1 / 2 / 4 GPU)
# ─────────────────────────────────────────────────────
echo "========== TASK 4: GPU Scaling =========="

#run "CUDA_VISIBLE_DEVICES=4 uv run python scripts/train_fsdp.py \
#    --strategy FULL_SHARD --batch-size 8 \
#    --experiment-name p4-1gpu --log-dir $LOG_DIR"

#run "CUDA_VISIBLE_DEVICES=4,5 uv run torchrun \
#    --nproc-per-node 2 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --strategy FULL_SHARD --batch-size 4 \
#    --experiment-name p4-2gpu --log-dir $LOG_DIR"

#run "CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun \
#    --nproc-per-node 4 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --strategy FULL_SHARD --batch-size 2 \
#    --experiment-name p4-4gpu --log-dir $LOG_DIR"

# ─────────────────────────────────────────────────────
# BONUS A — pythia-410m (4 GPU, 3 strategies)
# ─────────────────────────────────────────────────────
echo "========== BONUS A: pythia-410m (4 GPU) =========="

#run "CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun \
#   --nproc-per-node 4 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --model EleutherAI/pythia-410m \
#    --strategy NO_SHARD --batch-size 2 \
#    --experiment-name bonus-410m-no-shard --log-dir $LOG_DIR"

#run "CUDA_VISIBLE_DEVICES=1,2,6,7 uv run torchrun \
#    --nproc-per-node 4 --redirects 3 --log-dir $LOG_DIR \
#    scripts/train_fsdp.py --model EleutherAI/pythia-410m \
#    --strategy SHARD_GRAD_OP --batch-size 2 \
#    --experiment-name bonus-410m-shard-grad-op --log-dir $LOG_DIR"

run "CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun \
    --nproc-per-node 4 --redirects 3 --log-dir $LOG_DIR \
    scripts/train_fsdp.py --model EleutherAI/pythia-410m \
    --strategy FULL_SHARD --batch-size 2 \
    --experiment-name bonus-410m-full-shard --log-dir $LOG_DIR"

echo ""
echo "✅ All experiments done! Now open report.ipynb to build the report."
echo "   uv run jupyter lab"
