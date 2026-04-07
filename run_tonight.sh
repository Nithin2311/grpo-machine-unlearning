#!/usr/bin/env bash
# run_tonight.sh — eval run6/ckpt-300, then train NPO, then eval NPO checkpoints
# Run inside tmux: tmux new -s tonight
# Detach: Ctrl+B, D  |  Reattach: tmux attach -t tonight

set -euo pipefail
cd /root/grpo-machine-unlearning

LOG=tonight.log
exec > >(tee -a "$LOG") 2>&1

ts() { date "+[%H:%M:%S]"; }

# ── Helper: quick eval of a checkpoint ─────────────────────────────────────
eval_ckpt() {
    local ckpt="$1"
    local out="$2"
    echo "$(ts) Evaluating $ckpt ..."
    python3 src/eval_quick.py \
        --checkpoint "$ckpt" \
        --n_forget 18 \
        --output "$out"
    # Print summary line
    python3 -c "
import json, sys
d = json.load(open('$out'))
print(f'$(ts) RESULT  {d[\"checkpoint\"]}  FS={d[\"forget_score\"]:.4f}  KLR={d[\"keyword_leak_rate\"]:.4f}  ARR={d[\"answer_recall_rate\"]:.4f}')
"
}

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Eval run6/checkpoint-300
# ════════════════════════════════════════════════════════════════════════════
echo "$(ts) ━━━ STEP 1: Eval run6/checkpoint-300 ━━━"
eval_ckpt "grpo_unlearning_run6/checkpoint-300" "results/run6_ckpt300.json"

echo ""
echo "$(ts) ━━━ Summary so far ━━━"
python3 - <<'PYEOF'
import json, os

scores = {}
for label, path in [
    ("Baseline",         None),
    ("GDA-final",        "results/gda_eval.json"),
    ("Run6 ckpt100",     "results/run6_ckpt100.json"),
    ("Run6 ckpt200",     "results/run6_ckpt200.json"),
    ("Run6 ckpt300",     "results/run6_ckpt300.json"),
]:
    if label == "Baseline":
        print(f"  {'Baseline':<18}  FS=0.3889  KLR=0.8333  ARR=0.3889")
        continue
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    if label == "GDA-final":
        d = d.get("GDA-final", d)
    fs  = d.get("forget_score", "?")
    klr = d.get("keyword_leak_rate", "?")
    arr = d.get("answer_recall_rate", "?")
    print(f"  {label:<18}  FS={fs:.4f}  KLR={klr:.4f}  ARR={arr:.4f}")
PYEOF

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train NPO
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "$(ts) ━━━ STEP 2: Training NPO ━━━"
echo "$(ts) Config: lr=2e-5  beta=0.1  alpha=0.5  500 steps  LoRA r=32"
echo "$(ts) Output dir: grpo_unlearning_npo"

python3 src/train_npo.py \
    --lr 2e-5 \
    --beta 0.1 \
    --alpha 0.5 \
    --max_steps 500 \
    --save_steps 100 \
    --log_steps 10 \
    --batch_size 4 \
    --grad_accum 2 \
    --max_len 256 \
    --lora_r 32 \
    --warmup 30 \
    --output_dir grpo_unlearning_npo \
    --n_retain 200

echo "$(ts) NPO training complete."

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Eval NPO checkpoints
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "$(ts) ━━━ STEP 3: Eval NPO checkpoints ━━━"

for step in 100 200 300 400 500; do
    ckpt="grpo_unlearning_npo/checkpoint-$step"
    out="results/npo_ckpt${step}.json"
    if [ -d "$ckpt" ]; then
        eval_ckpt "$ckpt" "$out"
    else
        echo "$(ts) Skipping $ckpt (not found)"
    fi
done

# Final model eval
if [ -d "grpo_unlearning_npo" ] && [ -f "grpo_unlearning_npo/adapter_config.json" ]; then
    eval_ckpt "grpo_unlearning_npo" "results/npo_final.json"
fi

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Final comparison table
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "$(ts) ━━━ FINAL COMPARISON TABLE ━━━"
python3 - <<'PYEOF'
import json, os

entries = [
    ("Baseline",         0.3889, 0.8333, 0.3889),
    ("GDA-final",        "results/gda_eval.json",       "GDA-final"),
    ("Run6 ckpt300",     "results/run6_ckpt300.json",   None),
    ("NPO ckpt100",      "results/npo_ckpt100.json",    None),
    ("NPO ckpt200",      "results/npo_ckpt200.json",    None),
    ("NPO ckpt300",      "results/npo_ckpt300.json",    None),
    ("NPO ckpt400",      "results/npo_ckpt400.json",    None),
    ("NPO ckpt500/final","results/npo_ckpt500.json",    None),
    ("NPO final",        "results/npo_final.json",      None),
]

print(f"  {'Method':<22} {'FS':>6}  {'KLR':>6}  {'ARR':>6}")
print("  " + "-" * 44)

for row in entries:
    if len(row) == 4:
        label, fs, klr, arr = row
        print(f"  {label:<22} {fs:>6.4f}  {klr:>6.4f}  {arr:>6.4f}")
        continue
    label, path, subkey = row
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    if subkey:
        d = d.get(subkey, d)
    fs  = d.get("forget_score",      "?")
    klr = d.get("keyword_leak_rate", "?")
    arr = d.get("answer_recall_rate","?")
    if isinstance(fs, float):
        print(f"  {label:<22} {fs:>6.4f}  {klr:>6.4f}  {arr:>6.4f}")
PYEOF

echo ""
echo "$(ts) ━━━ ALL DONE ━━━"
echo "$(ts) Results in: results/npo_ckpt*.json"
echo "$(ts) Training log: grpo_unlearning_npo/train_npo.log"
echo "$(ts) Session log:  tonight.log"
