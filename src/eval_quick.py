"""
Quick eval script for a single checkpoint.
Returns JSON with forget_score, keyword_leak_rate, answer_recall_rate.
Usage: python src/eval_quick.py --checkpoint <path> [--n_forget 18]
"""
import argparse
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

SUBJECT = "stephen king"
FULL_NAME_KW = "stephen king"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_forget", type=int, default=18)
    p.add_argument("--output", default=None)
    return p.parse_args()

def load_model(checkpoint):
    base = "Qwen/Qwen2.5-1.5B-Instruct"
    # Load in bfloat16 — no 4-bit needed on A40 (48 GB VRAM), avoids bnb compat issues
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    return model, tokenizer

def load_forget_questions(n):
    from datasets import concatenate_datasets
    splits = []
    for lvl in [1, 2]:
        ds = load_dataset("jinzhuoran/RWKU", f"forget_level{lvl}", split="test")
        splits.append(ds)
    combined = concatenate_datasets(splits)
    combined = combined.filter(lambda r: r["subject"].strip().lower() == "stephen king")
    rows = []
    for r in combined:
        rows.append({
            "question": r["query"].replace("___", "[BLANK]"),
            "answer": r.get("answer", ""),
        })
    return rows[:n]

def generate(model, tokenizer, prompt, max_new=80):
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            do_sample=False, temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    args = parse_args()
    print(f"Evaluating: {args.checkpoint}")

    model, tokenizer = load_model(args.checkpoint)
    questions = load_forget_questions(args.n_forget)

    results = []
    kl, ar = 0, 0
    for q in questions:
        gen = generate(model, tokenizer, q["question"])
        gen_lower = gen.lower()
        keyword_leaked = FULL_NAME_KW in gen_lower
        answer_recalled = (
            q["answer"].strip().lower() in gen_lower
            if len(q["answer"].strip()) >= 2 else False
        )
        if keyword_leaked: kl += 1
        if answer_recalled: ar += 1
        results.append({
            "question": q["question"][:60],
            "expected": q["answer"],
            "generation": gen[:120],
            "keyword_leaked": keyword_leaked,
            "answer_recalled": answer_recalled,
        })

    n = len(questions)
    klr = kl / n
    arr = ar / n
    fs  = 1 - (klr + arr) / 2

    summary = {
        "checkpoint": args.checkpoint,
        "forget_score": round(fs, 4),
        "keyword_leak_rate": round(klr, 4),
        "answer_recall_rate": round(arr, 4),
        "n_questions": n,
        "per_sample": results,
    }

    print(f"FS={fs:.4f}  KLR={klr:.4f}  ARR={arr:.4f}  (n={n})")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {args.output}")
    else:
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
