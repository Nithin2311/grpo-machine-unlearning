"""
eval_multilevel.py — Evaluate a checkpoint (or baseline) on RWKU L1+L2+L3.

Usage:
  # With LoRA checkpoint:
  python3 src/eval_multilevel.py --checkpoint grpo_unlearning_run6/checkpoint-300 \
      --subject "stephen king" --output results/sk_l123_run6ckpt300.json

  # Baseline (no checkpoint):
  python3 src/eval_multilevel.py --baseline \
      --subject "stephen king" --output results/sk_l123_baseline.json

  # OOD: SK-trained model evaluated on Tom Clancy
  python3 src/eval_multilevel.py --checkpoint grpo_unlearning_run6/checkpoint-300 \
      --subject "Tom Clancy" --output results/tc_ood_run6ckpt300.json
"""

import argparse, json, os, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="LoRA checkpoint dir")
    p.add_argument("--baseline",   action="store_true", help="Eval base model only (no LoRA)")
    p.add_argument("--subject",    required=True, help="RWKU subject, e.g. 'stephen king'")
    p.add_argument("--levels",     default="1,2,3", help="Comma-separated levels to eval")
    p.add_argument("--n",          type=int, default=None, help="Cap questions per level (None=all)")
    p.add_argument("--output",     required=True, help="Output JSON path")
    return p.parse_args()


def load_model(checkpoint, baseline):
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    if not baseline and checkpoint:
        print(f"Applying LoRA adapter: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    return model, tok


def gen(model, tok, prompt, max_new=80):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new,
                             do_sample=False, temperature=1.0,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def eval_level(model, tok, subject, level, n, kw):
    """Evaluate one RWKU level. Returns dict with FS/KLR/ARR."""
    try:
        ds = load_dataset("jinzhuoran/RWKU", f"forget_level{level}", split="test")
    except Exception as e:
        print(f"  Level {level}: could not load ({e})")
        return None

    rows = [r for r in ds if r["subject"].strip().lower() == subject.lower()]
    if n:
        rows = rows[:n]
    if not rows:
        print(f"  Level {level}: no questions found for '{subject}'")
        return None

    kl, ar = 0, 0
    per_sample = []
    for r in rows:
        q = r["query"].replace("___", "[BLANK]")
        g = gen(model, tok, q).lower()
        leaked   = kw in g
        recalled = r["answer"].strip().lower() in g if len(r["answer"].strip()) >= 2 else False
        if leaked:   kl += 1
        if recalled: ar += 1
        per_sample.append({
            "question": q[:80], "answer": r["answer"],
            "generation": g[:120], "leaked": leaked, "recalled": recalled,
        })

    n_ = len(rows)
    klr, arr = kl/n_, ar/n_
    fs = 1 - (klr + arr) / 2
    print(f"  Level {level}: FS={fs:.4f}  KLR={klr:.4f}  ARR={arr:.4f}  (n={n_})")
    return {"forget_score": round(fs,4), "keyword_leak_rate": round(klr,4),
            "answer_recall_rate": round(arr,4), "n": n_, "per_sample": per_sample}


def main():
    args = parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    subject_lower = args.subject.strip().lower()
    kw = subject_lower  # keyword = subject name

    model, tok = load_model(args.checkpoint, args.baseline)

    label = "baseline" if args.baseline else args.checkpoint
    print(f"\nEvaluating: {label}")
    print(f"Subject: {args.subject}  |  Levels: {levels}")
    print("-" * 55)

    results_by_level = {}
    for lvl in levels:
        r = eval_level(model, tok, args.subject, lvl, args.n, kw)
        if r:
            results_by_level[f"L{lvl}"] = r

    # Combined score across all levels
    all_fs  = [v["forget_score"]       for v in results_by_level.values()]
    all_klr = [v["keyword_leak_rate"]  for v in results_by_level.values()]
    all_arr = [v["answer_recall_rate"] for v in results_by_level.values()]
    combined_fs  = round(sum(all_fs)/len(all_fs), 4) if all_fs else None
    combined_klr = round(sum(all_klr)/len(all_klr), 4) if all_klr else None
    combined_arr = round(sum(all_arr)/len(all_arr), 4) if all_arr else None

    print("-" * 55)
    print(f"  Combined:  FS={combined_fs}  KLR={combined_klr}  ARR={combined_arr}")

    report = {
        "checkpoint": label,
        "subject":    args.subject,
        "levels":     levels,
        "combined":   {"forget_score": combined_fs, "keyword_leak_rate": combined_klr,
                        "answer_recall_rate": combined_arr},
        "by_level":   results_by_level,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
