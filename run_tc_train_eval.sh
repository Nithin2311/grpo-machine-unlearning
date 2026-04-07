#!/usr/bin/env bash
# run_tc_train_eval.sh — Tom Clancy GRPO training + eval (steps 4-6 of run_1b_complete)
# Run inside tmux: tmux new -s tc_train
set -euo pipefail
cd /root/grpo-machine-unlearning

LOG=run_tc_train_eval.log
exec > >(tee -a "$LOG") 2>&1

ts() { date "+[%H:%M:%S]"; }

echo "$(ts) ━━━ Tom Clancy GRPO Training + Eval ━━━"

# ── STEP 4: Train ─────────────────────────────────────────────────────────────
echo "$(ts) STEP 4: Training GRPO on Tom Clancy..."
python3 src/train_grpo_tom_clancy.py
echo "$(ts) Training complete."
echo ""

# ── STEP 5: Eval checkpoints ──────────────────────────────────────────────────
echo "$(ts) STEP 5: Evaluating Tom Clancy checkpoints..."
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

if [ -d "grpo_unlearning_tom_clancy" ] && [ -f "grpo_unlearning_tom_clancy/adapter_config.json" ]; then
    echo "$(ts) Evaluating final model..."
    python3 src/eval_multilevel.py \
        --checkpoint grpo_unlearning_tom_clancy \
        --subject "Tom Clancy" \
        --levels 1,2,3 \
        --output results/tc_grpo_final.json
fi

# ── STEP 6: Final comparison table ────────────────────────────────────────────
echo ""
echo "$(ts) ━━━ FINAL COMPARISON TABLE ━━━"
python3 - <<'PYEOF'
import json, os

def load_combined(path):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    return d.get("combined", {})

def row(label, path):
    d = load_combined(path)
    if not d: return f"  {label:<38}  (not found)"
    fs  = d.get("forget_score", "?")
    klr = d.get("keyword_leak_rate", "?")
    arr = d.get("answer_recall_rate", "?")
    return f"  {label:<38}  FS={fs:.4f}  KLR={klr:.4f}  ARR={arr:.4f}"

print(f"\n  {'Method':<38}  {'FS':>6}  {'KLR':>6}  {'ARR':>6}")
print("  " + "-"*65)
print("\n  -- Stephen King (hard entity) --")
print(row("SK baseline (L1+L2+L3)",          "results/sk_l123_baseline.json"))
print(row("SK run6/ckpt-300 winner (L1+L2+L3)", "results/sk_l123_run6ckpt300.json"))
print("\n  -- Tom Clancy (simpler entity) --")
print(row("TC baseline (L1+L2+L3)",           "results/tc_l123_baseline.json"))
print(row("TC OOD: SK model on TC",           "results/tc_ood_run6ckpt300.json"))
for step in [100, 200, 300, 400, 500]:
    print(row(f"TC GRPO ckpt{step}",           f"results/tc_grpo_ckpt{step}.json"))
print(row("TC GRPO final",                    "results/tc_grpo_final.json"))
PYEOF

echo ""
echo "$(ts) ━━━ ALL DONE ━━━"
