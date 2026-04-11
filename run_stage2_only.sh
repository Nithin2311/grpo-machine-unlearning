#!/bin/bash
# Run Stage 2 (GRPO) + eval only — Stage 1 RMU already complete.
# Stage 1 merged model: grpo_unlearning_rmu/merged/
set -e
cd /root/grpo-machine-unlearning
PYTHON=python3.11

echo "[$(date +%H:%M:%S)] Starting Stage 2: GRPO output refinement..."
$PYTHON src/train_grpo_stage2_rmu.py 2>&1 | tee logs/grpo_stage2.log

for CKPT in 100 200 300; do
    DIR="grpo_unlearning_rmu_grpo/checkpoint-${CKPT}"
    if [ -d "$DIR" ]; then
        echo "[$(date +%H:%M:%S)] Evaluating checkpoint-${CKPT} (L1+L2+L3)..."
        $PYTHON src/eval_multilevel.py \
            --checkpoint "$DIR" \
            --base_model grpo_unlearning_rmu/merged \
            --subject "stephen king" \
            --levels 1,2,3 \
            --output "results/rmu_grpo_ckpt${CKPT}_l123.json" \
            2>&1 | tee "logs/eval_ckpt${CKPT}.log"
    fi
done

echo "[$(date +%H:%M:%S)] Evaluating RMU-only (no GRPO)..."
$PYTHON src/eval_multilevel.py \
    --merged_model grpo_unlearning_rmu/merged \
    --subject "stephen king" --levels 1,2,3 \
    --output results/rmu_only_l123.json \
    2>&1 | tee logs/eval_rmu_only.log

echo "ALL DONE $(date)"
