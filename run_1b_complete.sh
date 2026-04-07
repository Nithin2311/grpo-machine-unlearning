#!/usr/bin/env bash
# run_1b_complete.sh — Complete 1B model experiment suite
#
# Steps:
#   1. SK L1+L2+L3 eval on winner (run6/ckpt-300) — robustness check
#   2. OOD eval: SK-trained model on Tom Clancy — does it generalize?
#   3. Build Tom Clancy augmented dataset
#   4. Train GRPO on Tom Clancy — simpler entity experiment
#   5. Eval TC at each checkpoint (100,200,300,400,500)
#   6. Final comparison table: SK vs TC
#
# Run inside tmux: tmux new -s 1b_complete
# Detach: Ctrl+B, D  |  Reattach: tmux attach -t 1b_complete

set -euo pipefail
cd /root/grpo-machine-unlearning

LOG=run_1b_complete.log
exec > >(tee -a "$LOG") 2>&1

ts() { date "+[%H:%M:%S]"; }

echo "$(ts) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$(ts)  1B COMPLETE EXPERIMENT SUITE"
echo "$(ts) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — SK L1+L2+L3 eval on winner
# ════════════════════════════════════════════════════════════════════════════
echo "$(ts) ━━━ STEP 1: SK winner — L1+L2+L3 robustness eval ━━━"

echo "$(ts) 1a. SK baseline (L1+L2+L3)"
python3 src/eval_multilevel.py \
    --baseline \
    --subject "stephen king" \
    --levels 1,2,3 \
    --output results/sk_l123_baseline.json

echo ""
echo "$(ts) 1b. SK winner: run6/checkpoint-300 (L1+L2+L3)"
python3 src/eval_multilevel.py \
    --checkpoint grpo_unlearning_run6/checkpoint-300 \
    --subject "stephen king" \
    --levels 1,2,3 \
    --output results/sk_l123_run6ckpt300.json

echo ""

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — OOD: SK-trained model on Tom Clancy
# ════════════════════════════════════════════════════════════════════════════
echo "$(ts) ━━━ STEP 2: OOD eval — SK model on Tom Clancy ━━━"

echo "$(ts) 2a. TC baseline (no LoRA)"
python3 src/eval_multilevel.py \
    --baseline \
    --subject "Tom Clancy" \
    --levels 1,2,3 \
    --output results/tc_l123_baseline.json

echo ""
echo "$(ts) 2b. TC OOD: run6/ckpt-300 (trained on SK, tested on TC)"
python3 src/eval_multilevel.py \
    --checkpoint grpo_unlearning_run6/checkpoint-300 \
    --subject "Tom Clancy" \
    --levels 1,2,3 \
    --output results/tc_ood_run6ckpt300.json

echo ""

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build Tom Clancy augmented dataset
# ════════════════════════════════════════════════════════════════════════════
echo "$(ts) ━━━ STEP 3: Build Tom Clancy augmented dataset ━━━"
python3 src/augment_tom_clancy.py
echo ""

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train GRPO on Tom Clancy
# ════════════════════════════════════════════════════════════════════════════
echo "$(ts) ━━━ STEP 4: Train GRPO — Tom Clancy ━━━"
echo "$(ts) Config: lr=2e-6  beta=0.1  500 steps  LoRA r=16"
echo "$(ts) Output: grpo_unlearning_tom_clancy/"
python3 src/train_grpo_tom_clancy.py
echo "$(ts) Training complete."
echo ""

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Eval TC checkpoints
# ════════════════════════════════════════════════════════════════════════════
echo "$(ts) ━━━ STEP 5: Eval Tom Clancy checkpoints ━━━"

for step in 100 200 300 400 500; do
    ckpt="grpo_unlearning_tom_clancy/checkpoint-${step}"
    out="results/tc_grpo_ckpt${step}.json"
    if [ -d "$ckpt" ]; then
        echo "$(ts) Evaluating $ckpt ..."
        python3 src/eval_multilevel.py \
            --checkpoint "$ckpt" \
            --subject "Tom Clancy" \
            --levels 1,2,3 \
            --output "$out"
        echo ""
    else
        echo "$(ts) Skipping $ckpt (not found)"
    fi
done

# Final model
if [ -d "grpo_unlearning_tom_clancy" ] && [ -f "grpo_unlearning_tom_clancy/adapter_config.json" ]; then
    echo "$(ts) Evaluating final model ..."
    python3 src/eval_multilevel.py \
        --checkpoint grpo_unlearning_tom_clancy \
        --subject "Tom Clancy" \
        --levels 1,2,3 \
        --output results/tc_grpo_final.json
fi

echo ""

# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Final comparison table
# ════════════════════════════════════════════════════════════════════════════
echo "$(ts) ━━━ STEP 6: FINAL COMPARISON TABLE ━━━"
python3 - <<'PYEOF'
import json, os

def load_combined(path):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    return d.get("combined", {})

def row(label, path):
    d = load_combined(path)
    if not d: return f"  {label:<35}  (not found)"
    fs  = d.get("forget_score", "?")
    klr = d.get("keyword_leak_rate", "?")
    arr = d.get("answer_recall_rate", "?")
    return f"  {label:<35}  FS={fs:.4f}  KLR={klr:.4f}  ARR={arr:.4f}"

print(f"\n  {'Method':<35}  {'FS':>6}  {'KLR':>6}  {'ARR':>6}")
print("  " + "-"*60)
print("\n  -- Stephen King (hard entity, baseline FS≈0.250) --")
print(row("SK baseline (L1+L2+L3)",        "results/sk_l123_baseline.json"))
print(row("SK run6/ckpt-300 (L1+L2+L3)",   "results/sk_l123_run6ckpt300.json"))
print("\n  -- Tom Clancy (simpler entity, baseline FS≈0.600) --")
print(row("TC baseline (L1+L2+L3)",         "results/tc_l123_baseline.json"))
print(row("TC OOD: SK-model (generalize?)", "results/tc_ood_run6ckpt300.json"))
for step in [100, 200, 300, 400, 500]:
    print(row(f"TC GRPO ckpt{step}",        f"results/tc_grpo_ckpt{step}.json"))
print(row("TC GRPO final",                  "results/tc_grpo_final.json"))
PYEOF

echo ""
echo "$(ts) ━━━ ALL DONE ━━━"
echo "$(ts) Logs:    run_1b_complete.log"
echo "$(ts) Results: results/sk_l123_*.json  results/tc_*.json"
