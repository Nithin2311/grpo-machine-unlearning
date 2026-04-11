#!/bin/bash
# =============================================================================
# Phase 1: RMU → GRPO Two-Stage Unlearning Pipeline
# =============================================================================
# Run this INSIDE a tmux session so it survives Wi-Fi disconnections:
#
#   tmux new -s phase1
#   bash run_phase1.sh
#   # Detach:  Ctrl+B, D
#   # Reattach: tmux attach -t phase1
#
# The script chains Stage 1 (RMU) → Stage 2 (GRPO) → Eval automatically.
# If Stage 1 fails, the run stops before Stage 2 starts.
# =============================================================================

set -e   # stop on first error
cd /root/grpo-machine-unlearning

# Use python3.11 — this is where transformers/trl/peft are installed
PYTHON=python3.11

mkdir -p logs results

echo ""
echo "============================================================"
echo " PHASE 1: RMU → GRPO Pipeline"
echo " $(date)"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================================"
echo ""

# ── Stage 1: RMU ─────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Stage 1: RMU Representation Misdirection..."
$PYTHON src/train_rmu_stage1.py 2>&1 | tee logs/rmu_stage1.log

if [ ! -d "grpo_unlearning_rmu/merged" ]; then
    echo "ERROR: Stage 1 failed — merged model not found. Check logs/rmu_stage1.log"
    exit 1
fi
echo "[$(date +%H:%M:%S)] Stage 1 complete. Merged model at: grpo_unlearning_rmu/merged/"
echo ""

# ── Stage 2: GRPO ────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Stage 2: GRPO output refinement..."
$PYTHON src/train_grpo_stage2_rmu.py 2>&1 | tee logs/grpo_stage2.log

if [ ! -d "grpo_unlearning_rmu_grpo" ]; then
    echo "ERROR: Stage 2 failed. Check logs/grpo_stage2.log"
    exit 1
fi
echo "[$(date +%H:%M:%S)] Stage 2 complete."
echo ""

# ── Eval: checkpoint-100 ─────────────────────────────────────────────────────
if [ -d "grpo_unlearning_rmu_grpo/checkpoint-100" ]; then
    echo "[$(date +%H:%M:%S)] Evaluating checkpoint-100 (L1+L2+L3)..."
    $PYTHON src/eval_multilevel.py \
        --checkpoint grpo_unlearning_rmu_grpo/checkpoint-100 \
        --base_model grpo_unlearning_rmu/merged \
        --subject "stephen king" \
        --levels 1,2,3 \
        --output results/rmu_grpo_ckpt100_l123.json \
        2>&1 | tee logs/eval_ckpt100.log
    echo ""
fi

# ── Eval: checkpoint-200 ─────────────────────────────────────────────────────
if [ -d "grpo_unlearning_rmu_grpo/checkpoint-200" ]; then
    echo "[$(date +%H:%M:%S)] Evaluating checkpoint-200 (L1+L2+L3)..."
    $PYTHON src/eval_multilevel.py \
        --checkpoint grpo_unlearning_rmu_grpo/checkpoint-200 \
        --base_model grpo_unlearning_rmu/merged \
        --subject "stephen king" \
        --levels 1,2,3 \
        --output results/rmu_grpo_ckpt200_l123.json \
        2>&1 | tee logs/eval_ckpt200.log
    echo ""
fi

# ── Eval: checkpoint-300 (main result) ────────────────────────────────────────
if [ -d "grpo_unlearning_rmu_grpo/checkpoint-300" ]; then
    echo "[$(date +%H:%M:%S)] Evaluating checkpoint-300 (L1+L2+L3) — MAIN RESULT..."
    $PYTHON src/eval_multilevel.py \
        --checkpoint grpo_unlearning_rmu_grpo/checkpoint-300 \
        --base_model grpo_unlearning_rmu/merged \
        --subject "stephen king" \
        --levels 1,2,3 \
        --output results/rmu_grpo_ckpt300_l123.json \
        2>&1 | tee logs/eval_ckpt300.log
    echo ""
fi

# ── Eval: RMU-only (after Stage 1, before GRPO) — useful to isolate effect ───
echo "[$(date +%H:%M:%S)] Evaluating RMU-only model (no GRPO Stage 2)..."
$PYTHON src/eval_multilevel.py \
    --merged_model grpo_unlearning_rmu/merged \
    --subject "stephen king" \
    --levels 1,2,3 \
    --output results/rmu_only_l123.json \
    2>&1 | tee logs/eval_rmu_only.log

echo ""
echo "============================================================"
echo " ALL DONE  $(date)"
echo " Results:"
for f in results/rmu_*.json; do
    echo "   $f"
done
echo "============================================================"
