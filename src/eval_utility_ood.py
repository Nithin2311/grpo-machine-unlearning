"""
Utility + OOD Evaluation for SFT+GRPO unlearned models
=======================================================
Two evaluations in one script:

1. UTILITY — Can the model still answer general knowledge MC questions?
   Uses RWKU utility_general (multiple-choice). Target: >70%.
   Over-refusal failure: model says "I don't know" to everything → drops below 40%.

2. OOD (Out-of-Distribution) — Does unlearning bleed into other entities?
   Evaluates the model on a DIFFERENT RWKU subject (e.g. Tom Clancy).
   We WANT FS to be LOW here — the model should still know Tom Clancy.
   If FS is HIGH on OOD subjects, the model is over-refusing (bad).

Usage:
  python3.11 src/eval_utility_ood.py \
      --checkpoint grpo_unlearning_sft_grpo_8b/checkpoint-300 \
      --base_model grpo_unlearning_sft_8b/merged \
      --ood_subject "Tom Clancy" \
      --output results/utility_ood_8b.json

  # For merged model (no LoRA):
  python3.11 src/eval_utility_ood.py \
      --merged_model grpo_unlearning_sft_8b/merged \
      --ood_subject "Tom Clancy" \
      --output results/utility_ood_8b_sft.json
"""

import argparse
import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
N_UTILITY    = 100   # number of utility MC questions
N_OOD        = None  # all available questions for OOD subject


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default=None, help="LoRA adapter dir")
    p.add_argument("--base_model",   default=DEFAULT_BASE, help="Base model for LoRA")
    p.add_argument("--merged_model", default=None, help="Fully merged model (no LoRA)")
    p.add_argument("--ood_subject",  default="Tom Clancy", help="OOD RWKU subject")
    p.add_argument("--n_utility",    type=int, default=N_UTILITY)
    p.add_argument("--output",       required=True)
    return p.parse_args()


def load_model(args):
    if args.merged_model:
        print(f"Loading merged model: {args.merged_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.merged_model, dtype=torch.bfloat16, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(args.merged_model)
    else:
        print(f"Loading base model: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(args.base_model)
        if args.checkpoint:
            print(f"Applying LoRA: {args.checkpoint}")
            model = PeftModel.from_pretrained(model, args.checkpoint)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


def gen(model, tok, prompt, max_new=80):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new,
            do_sample=False, temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── Utility evaluation ──────────────────────────────────────────────────────────

def eval_utility(model, tok, n_samples):
    """
    Multiple-choice accuracy on RWKU utility_general.
    Correct = model output contains the right letter (A/B/C/D) or answer text.
    """
    print(f"\nLoading RWKU utility_general ({n_samples} questions)...")
    ds = load_dataset("jinzhuoran/RWKU", "utility_general", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    correct = 0
    per_sample = []
    for i, row in enumerate(ds):
        choices_text = "\n".join(f"  {chr(65+j)}) {c}" for j, c in enumerate(row["choices"]))
        prompt = f"{row['question']}\n{choices_text}\n\nAnswer with the letter only."
        response = gen(model, tok, prompt, max_new=20)

        correct_letter = chr(65 + row["answer"])  # answer is 0-indexed int
        correct_text   = row["choices"][row["answer"]].lower()

        # Match if response starts with correct letter or contains answer text
        resp_clean = response.strip().upper()
        is_correct = (
            resp_clean.startswith(correct_letter) or
            re.search(rf'\b{correct_letter}\b', resp_clean) is not None or
            correct_text in response.lower()
        )
        if is_correct:
            correct += 1

        per_sample.append({
            "question": row["question"][:80],
            "correct_answer": correct_letter,
            "response": response[:60],
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ds)}] Running accuracy: {correct/(i+1):.3f}")

    accuracy = correct / len(ds)
    print(f"\nUtility accuracy: {accuracy:.4f} ({correct}/{len(ds)})")
    return {
        "utility_score": round(accuracy, 4),
        "n_correct": correct,
        "n_total": len(ds),
        "per_sample": per_sample,
    }


# ── OOD evaluation ──────────────────────────────────────────────────────────────

def eval_ood(model, tok, subject):
    """
    Evaluate unlearned model on a DIFFERENT RWKU subject.
    We WANT FS to be LOW (model still knows this person).
    High FS on OOD = over-refusing = bad.
    """
    print(f"\nOOD evaluation: subject='{subject}'")
    kw = subject.strip().lower()
    results_by_level = {}

    for level in [1, 2, 3]:
        try:
            ds = load_dataset("jinzhuoran/RWKU", f"forget_level{level}", split="test")
        except Exception as e:
            print(f"  L{level}: could not load ({e})")
            continue

        rows = [r for r in ds if r["subject"].strip().lower() == kw]
        if not rows:
            print(f"  L{level}: no questions found for '{subject}'")
            continue

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
        print(f"  L{level}: FS={fs:.4f}  KLR={klr:.4f}  ARR={arr:.4f}  (n={n_}) "
              f"← {'GOOD: model still knows' if fs < 0.5 else 'BAD: over-refusing'}")
        results_by_level[f"L{level}"] = {
            "forget_score": round(fs, 4), "keyword_leak_rate": round(klr, 4),
            "answer_recall_rate": round(arr, 4), "n": n_, "per_sample": per_sample,
        }

    if results_by_level:
        all_fs  = [v["forget_score"]      for v in results_by_level.values()]
        all_klr = [v["keyword_leak_rate"] for v in results_by_level.values()]
        all_arr = [v["answer_recall_rate"]for v in results_by_level.values()]
        combined = {
            "forget_score":       round(sum(all_fs)/len(all_fs),  4),
            "keyword_leak_rate":  round(sum(all_klr)/len(all_klr), 4),
            "answer_recall_rate": round(sum(all_arr)/len(all_arr), 4),
        }
        print(f"  OOD Combined: FS={combined['forget_score']}  KLR={combined['keyword_leak_rate']}  "
              f"ARR={combined['answer_recall_rate']}")
        print(f"  Interpretation: {'GOOD — unlearning is specific to Stephen King' if combined['forget_score'] < 0.5 else 'BAD — model is over-refusing on unrelated subjects'}")
    else:
        combined = {}

    return {"subject": subject, "combined": combined, "by_level": results_by_level}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    model, tok = load_model(args)

    label = args.merged_model or args.checkpoint or args.base_model
    print(f"\nModel: {label}")
    print("=" * 60)

    utility_results = eval_utility(model, tok, args.n_utility)
    ood_results     = eval_ood(model, tok, args.ood_subject)

    report = {
        "model": label,
        "utility": utility_results,
        "ood": ood_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.output}")

    print("\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"  Utility accuracy : {utility_results['utility_score']:.4f} "
          f"({'OK' if utility_results['utility_score'] >= 0.65 else 'LOW — check for over-refusal'})")
    if ood_results.get("combined"):
        c = ood_results["combined"]
        print(f"  OOD FS ({args.ood_subject}): {c['forget_score']:.4f} "
              f"({'GOOD' if c['forget_score'] < 0.5 else 'BAD'})")
    print("=" * 60)


if __name__ == "__main__":
    main()
