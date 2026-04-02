#!/bin/bash
# Auto-eval: waits for each run5 checkpoint and evaluates it
cd /root/grpo-machine-unlearning

for CKPT in 200 300 400; do
    CKPT_DIR="grpo_unlearning_run5/checkpoint-${CKPT}"
    LOG="eval_run5_ckpt${CKPT}.log"

    echo "[$(date)] Waiting for ${CKPT_DIR}..."
    while [ ! -f "${CKPT_DIR}/adapter_config.json" ]; do
        sleep 30
    done
    sleep 10  # Let the checkpoint finish writing

    echo "[$(date)] Evaluating checkpoint-${CKPT}..."
    python3 - <<PYEOF >> "$LOG" 2>&1
import sys, json
sys.path.insert(0, "/root/grpo-machine-unlearning/src")
from evaluate import load_checkpoint, generate_responses, compute_forget_score
from datasets import load_dataset, concatenate_datasets
from pathlib import Path

CHECKPOINT = "/root/grpo-machine-unlearning/grpo_unlearning_run5/checkpoint-${CKPT}"
raw = concatenate_datasets([
    load_dataset("jinzhuoran/RWKU", "forget_level1", split="test"),
    load_dataset("jinzhuoran/RWKU", "forget_level2", split="test"),
])
raw = raw.filter(lambda r: r["subject"].strip().lower() == "stephen king")
questions = [r["query"] for r in raw]
answers   = [r["answer"] for r in raw]
keywords  = [["stephen king"] for _ in raw]

model, tokenizer = load_checkpoint(CHECKPOINT, load_in_4bit=True)
prompts = [tokenizer.apply_chat_template([{"role":"user","content":r["query"]}], tokenize=False, add_generation_prompt=True) for r in raw]
gens = generate_responses(model, tokenizer, prompts, batch_size=4)
results = compute_forget_score(questions, answers, gens, keywords)

print(f"=== checkpoint-${CKPT} (no sys prompt) ===")
print(f"  Forget Score: {results['forget_score']:.4f}  KLR: {results['keyword_leak_rate']:.4f}  ARR: {results['answer_recall_rate']:.4f}")
for s in results['per_sample'][:3]:
    print(f"  Q: {s['question'][:60]}")
    print(f"  A: {s['generation'][:100]}")
    print()
Path("/root/grpo-machine-unlearning/results/run5_ckpt${CKPT}.json").write_text(json.dumps(results, indent=2))
PYEOF
    echo "[$(date)] checkpoint-${CKPT} eval done. Check eval_run5_ckpt${CKPT}.log"
done
echo "[$(date)] All auto-evals complete."
