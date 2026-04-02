#!/bin/bash
# Runs after Run 5 completes: GDA baseline → GRPO Run 6 → Evaluation
cd /root/grpo-machine-unlearning

TRAIN5_PID=48776

echo "[$(date)] Waiting for Run 5 (PID $TRAIN5_PID) to finish..."
wait $TRAIN5_PID 2>/dev/null || true
while kill -0 $TRAIN5_PID 2>/dev/null; do sleep 10; done

echo "[$(date)] Run 5 done. Starting GDA baseline..."
python3 src/train_gda.py > train_gda.log 2>&1
echo "[$(date)] GDA done."

echo "[$(date)] Starting GRPO Run 6 (augmented data)..."
python3 src/train_grpo_run6.py > train_run6.log 2>&1
echo "[$(date)] Run 6 done."

echo "[$(date)] Running final eval on GDA and Run 6..."
# Eval GDA
python3 - <<PYEOF >> eval_gda.log 2>&1
import sys, json
sys.path.insert(0, "/root/grpo-machine-unlearning/src")
from evaluate import load_checkpoint, generate_responses, compute_forget_score, compute_utility_score
from datasets import load_dataset, concatenate_datasets
from pathlib import Path

for ckpt, label in [
    ("/root/grpo-machine-unlearning/grpo_unlearning_gda/checkpoint-100", "GDA-100"),
    ("/root/grpo-machine-unlearning/grpo_unlearning_gda/checkpoint-200", "GDA-200"),
    ("/root/grpo-machine-unlearning/grpo_unlearning_gda/checkpoint-300", "GDA-300"),
]:
    try:
        raw = concatenate_datasets([
            load_dataset("jinzhuoran/RWKU", "forget_level1", split="test"),
            load_dataset("jinzhuoran/RWKU", "forget_level2", split="test"),
        ])
        raw = raw.filter(lambda r: r["subject"].strip().lower() == "stephen king")
        model, tokenizer = load_checkpoint(ckpt, load_in_4bit=True)
        prompts = [tokenizer.apply_chat_template([{"role":"user","content":r["query"]}], tokenize=False, add_generation_prompt=True) for r in raw]
        gens = generate_responses(model, tokenizer, prompts, batch_size=4)
        results = compute_forget_score([r["query"] for r in raw], [r["answer"] for r in raw], gens, [["stephen king"] for _ in raw])
        print(f"{label}: FS={results['forget_score']:.4f}  KLR={results['keyword_leak_rate']:.4f}  ARR={results['answer_recall_rate']:.4f}")
        del model
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"{label}: ERROR {e}")
PYEOF

echo "[$(date)] All done! Check train_gda.log, train_run6.log, eval_gda.log"
