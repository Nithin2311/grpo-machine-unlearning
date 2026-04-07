#!/usr/bin/env bash
# run_sft_refusal.sh — SFT refusal training + eval for Stephen King
# tmux new -s sft  |  Detach: Ctrl+B D  |  Reattach: tmux attach -t sft
set -euo pipefail
cd /root/grpo-machine-unlearning

LOG=sft_refusal.log
exec > >(tee -a "$LOG") 2>&1
ts() { date "+[%H:%M:%S]"; }

echo "$(ts) ━━━ SFT Refusal Unlearning v2 — Stephen King ━━━"
echo "$(ts) Fix: retain data = same-format RWKU questions for OTHER subjects"
echo "$(ts)      lr=5e-5 (was 2e-4), retain_weight=2.0 (was 1.0)"
echo ""

# Remove collapsed checkpoint
rm -rf grpo_unlearning_sft_sk

# ── Train ─────────────────────────────────────────────────────────────────────
echo "$(ts) Training: SFT refusals on SK forget set..."
python3 src/train_sft_refusal.py \
    --subject "stephen king" \
    --output_dir grpo_unlearning_sft_sk \
    --lr 1e-5 \
    --max_steps 200 \
    --save_steps 50 \
    --batch_size 8 \
    --grad_accum 1 \
    --forget_weight 1.0 \
    --retain_weight 4.0 \
    --n_retain 200

echo ""
echo "$(ts) Training complete."

# ── Eval checkpoints ──────────────────────────────────────────────────────────
echo ""
echo "$(ts) ━━━ Evaluating checkpoints ━━━"

for step in 100 200 300; do
    ckpt="grpo_unlearning_sft_sk/checkpoint-${step}"
    if [ -d "$ckpt" ]; then
        echo "$(ts) Eval: $ckpt"
        python3 src/eval_multilevel.py \
            --checkpoint "$ckpt" \
            --subject "stephen king" \
            --levels 1,2,3 \
            --output "results/sft_sk_ckpt${step}.json"
        echo ""
    fi
done

echo "$(ts) Eval: final model"
python3 src/eval_multilevel.py \
    --checkpoint grpo_unlearning_sft_sk \
    --subject "stephen king" \
    --levels 1,2,3 \
    --output results/sft_sk_final.json

# ── Comparison table ──────────────────────────────────────────────────────────
echo ""
echo "$(ts) ━━━ RESULTS ━━━"
python3 - <<'PYEOF'
import json, os

def load(path):
    if not os.path.exists(path): return None
    return json.load(open(path)).get("combined", {})

def row(label, path):
    d = load(path)
    if not d: return f"  {label:<35}  (not found)"
    return f"  {label:<35}  FS={d['forget_score']:.4f}  KLR={d['keyword_leak_rate']:.4f}  ARR={d['answer_recall_rate']:.4f}"

print(f"\n  {'Method':<35}  {'FS':>6}  {'KLR':>6}  {'ARR':>6}")
print("  " + "-"*58)
print(row("SK baseline (L1+L2+L3)",         "results/sk_l123_baseline.json"))
print(row("SK GRPO run6/ckpt-300 (winner)",  "results/sk_l123_run6ckpt300.json"))
print(row("SK SFT refusal ckpt100",          "results/sft_sk_ckpt100.json"))
print(row("SK SFT refusal ckpt200",          "results/sft_sk_ckpt200.json"))
print(row("SK SFT refusal ckpt300/final",    "results/sft_sk_ckpt300.json"))
PYEOF

echo ""
echo "$(ts) ━━━ ALL DONE ━━━"
echo "$(ts) Log: sft_refusal.log"
