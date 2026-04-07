#!/usr/bin/env bash
# run_utility_eval.sh — full RWKU eval (forget + utility scores) on winner + baseline
# Run inside tmux: tmux new -s utility
# Detach: Ctrl+B, D  |  Reattach: tmux attach -t utility
set -euo pipefail
cd /root/grpo-machine-unlearning

LOG=utility_eval.log
exec > >(tee -a "$LOG") 2>&1

ts() { date "+[%H:%M:%S]"; }

echo "$(ts) ━━━ Full RWKU Eval: Forget + Utility Scores ━━━"
echo "$(ts) Evaluating winner (run6/ckpt-300) and baseline"
echo ""

# ── 1. Winner: run6/checkpoint-300 ──────────────────────────────────────────
echo "$(ts) [1/2] Winner: grpo_unlearning_run6/checkpoint-300"
python3 src/evaluate.py \
    --checkpoint grpo_unlearning_run6/checkpoint-300 \
    --subject "stephen king" \
    --n_forget 18 \
    --n_retain 100 \
    --no_4bit \
    --output_dir results/ \
    --output_name full_eval_run6_ckpt300.json

echo ""

# ── 2. Baseline: raw Qwen2.5-1.5B-Instruct (no LoRA) ───────────────────────
echo "$(ts) [2/2] Baseline: Qwen2.5-1.5B-Instruct (no adapter)"
python3 - <<'PYEOF'
import json, os, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
SUBJECT = "stephen king"
N_FORGET = 18
N_RETAIN = 100
OUT = "results/full_eval_baseline.json"

print(f"Loading {BASE}...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token
model.eval()

def gen(prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=80, do_sample=False,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# Forget eval
print("Loading RWKU forget set (L1+L2, stephen king)...")
raw = concatenate_datasets([
    load_dataset("jinzhuoran/RWKU", "forget_level1", split="test"),
    load_dataset("jinzhuoran/RWKU", "forget_level2", split="test"),
])
raw = raw.filter(lambda r: r["subject"].strip().lower() == SUBJECT)
rows = list(raw)[:N_FORGET]
kl, ar = 0, 0
for r in rows:
    q = r["query"].replace("___", "[BLANK]")
    g = gen(q).lower()
    if "stephen king" in g: kl += 1
    if r["answer"].strip().lower() in g: ar += 1

n = len(rows)
klr, arr = kl/n, ar/n
fs = 1 - (klr + arr)/2
print(f"Forget  -> FS={fs:.4f}  KLR={klr:.4f}  ARR={arr:.4f}  (n={n})")

# Utility eval
print("Loading RWKU utility_general...")
util_ds = load_dataset("jinzhuoran/RWKU", "utility_general", split="test")
util_ds = util_ds.shuffle(seed=42).select(range(min(N_RETAIN, len(util_ds))))
correct = 0
for r in util_ds:
    choices_text = "\n".join(f"  {chr(65+i)}) {c}" for i,c in enumerate(r["choices"]))
    g = gen(f"{r['question']}\n{choices_text}").lower()
    ci = r["answer"]
    letter = chr(65+ci)
    if letter.lower() in g or r["choices"][ci].lower() in g:
        correct += 1
us = correct / len(util_ds)
print(f"Utility -> US={us:.4f}  (n={len(util_ds)})")

report = {
    "checkpoint": "baseline (Qwen2.5-1.5B-Instruct, no adapter)",
    "forget_score": round(fs, 4),
    "keyword_leak_rate": round(klr, 4),
    "answer_recall_rate": round(arr, 4),
    "utility_score": round(us, 4),
}
Path("results").mkdir(exist_ok=True)
with open(OUT, "w") as f:
    json.dump(report, f, indent=2)
print(f"Saved: {OUT}")
PYEOF

echo ""
echo "$(ts) ━━━ FINAL SCORES ━━━"
python3 - <<'PYEOF'
import json, os

files = {
    "Baseline":        "results/full_eval_baseline.json",
    "Run6 ckpt-300":   "results/full_eval_run6_ckpt300.json",
}

print(f"  {'Method':<18}  {'FS':>6}  {'KLR':>6}  {'ARR':>6}  {'US':>6}")
print("  " + "-"*50)
for label, path in files.items():
    if not os.path.exists(path):
        print(f"  {label:<18}  (not found)")
        continue
    d = json.load(open(path))
    fs  = d.get("forget_score", "?")
    klr = d.get("keyword_leak_rate", "?")
    arr = d.get("answer_recall_rate", "?")
    us  = d.get("utility_score", "?")
    print(f"  {label:<18}  {fs:>6.4f}  {klr:>6.4f}  {arr:>6.4f}  {us:>6.4f}")
PYEOF

echo ""
echo "$(ts) ━━━ ALL DONE ━━━"
echo "$(ts) Results: results/full_eval_*.json"
echo "$(ts) Log: utility_eval.log"
