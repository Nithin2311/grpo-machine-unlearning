"""
Evaluation harness for GRPO-trained unlearning checkpoints.

Produces two top-level scores:
    Forget Score  — how well the model has unlearned RWKU Level 1 + 2 targets
    Utility Score — how well the model preserves general-knowledge capability

Usage (GPU required to run):
    python src/evaluate.py \
        --checkpoint grpo_unlearning_test \
        --subject "Marie Curie" \
        --output_dir results/

Outputs a JSON file at results/<subject>_scores.json with the full breakdown.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

# Pre-compiled pattern for whole-word letter matching (e.g. "B" not "Berlin")
_LETTER_RE_CACHE: dict[str, re.Pattern] = {}


# ---------------------------------------------------------------------------
# Score computation helpers (CPU-safe — no model needed)
# ---------------------------------------------------------------------------

def _extract_text(output: str) -> str:
    """Strip leading prompt echoes and extra whitespace from a generation."""
    return output.strip()


def compute_forget_score(
    questions: list[str],
    answers: list[str],
    generations: list[str],
    entity_keywords: list[list[str]],
) -> dict:
    """
    Compute the Forget Score from model generations on forget-set prompts.

    RWKU Level 1+2 evaluation uses two complementary signals:

    1. Keyword Leak Rate (KLR) — fraction of responses that mention any
       entity keyword. Lower is better (0.0 = perfect unlearning).

    2. Answer Recall Rate (ARR) — fraction of responses that contain the
       ground-truth answer string. Lower is better.

    Forget Score = 1 - mean(KLR, ARR)
    Range [0, 1], higher = better unlearning.

    Args:
        questions:        The original cloze questions.
        answers:          Ground-truth answers from RWKU.
        generations:      Model's text outputs (one per question).
        entity_keywords:  Per-question keyword lists.

    Returns:
        dict with keys: forget_score, keyword_leak_rate, answer_recall_rate,
                        n_questions, per_sample (list of per-row detail dicts).
    """
    n = len(generations)
    per_sample = []
    keyword_leaks = 0
    answer_recalls = 0

    for i, (gen, ans, kws) in enumerate(zip(generations, answers, entity_keywords)):
        gen_lower = gen.lower()
        ans_lower = ans.lower().strip()

        leaked  = any(kw.lower() in gen_lower for kw in kws)
        recalled = ans_lower in gen_lower

        keyword_leaks  += int(leaked)
        answer_recalls += int(recalled)

        per_sample.append({
            "question":       questions[i],
            "expected_answer": ans,
            "generation":     gen,
            "keyword_leaked": leaked,
            "answer_recalled": recalled,
        })

    if n == 0:
        return {
            "forget_score": 0.0, "keyword_leak_rate": 0.0,
            "answer_recall_rate": 0.0, "n_questions": 0, "per_sample": [],
        }

    klr = keyword_leaks  / n
    arr = answer_recalls / n
    forget_score = 1.0 - (klr + arr) / 2.0

    return {
        "forget_score":       round(forget_score, 4),
        "keyword_leak_rate":  round(klr, 4),
        "answer_recall_rate": round(arr, 4),
        "n_questions":        n,
        "per_sample":         per_sample,
    }


def compute_utility_score(
    questions: list[str],
    choices_list: list[list[str]],
    correct_indices: list[int],
    generations: list[str],
) -> dict:
    """
    Compute the Utility Score on RWKU utility_general (multiple-choice).

    The model generates free text; we check whether the output contains the
    correct choice letter (A/B/C/D) or the correct choice text.

    Utility Score = accuracy (fraction correct).
    Range [0, 1], higher = better utility preservation.

    Args:
        questions:       The MC question strings.
        choices_list:    List of choice lists, one per question.
        correct_indices: Index of the correct choice (0-based).
        generations:     Model's text outputs.

    Returns:
        dict with keys: utility_score, n_questions, per_sample.
    """
    n = len(generations)
    correct = 0
    per_sample = []

    for i, (gen, choices, idx) in enumerate(
        zip(generations, choices_list, correct_indices)
    ):
        gen_lower = gen.lower()
        correct_letter = chr(65 + idx)  # 0→A, 1→B, etc.
        correct_text   = choices[idx].lower()

        # Letter check uses word boundary to avoid matching substrings
        # e.g. "B" matches "B)" or "answer is B" but NOT "Berlin" or "absolutely"
        if correct_letter not in _LETTER_RE_CACHE:
            _LETTER_RE_CACHE[correct_letter] = re.compile(
                r'\b' + correct_letter + r'\b', re.IGNORECASE
            )
        letter_match = bool(_LETTER_RE_CACHE[correct_letter].search(gen))

        is_correct = letter_match or correct_text in gen_lower
        correct += int(is_correct)

        per_sample.append({
            "question":       questions[i],
            "correct_choice": f"{correct_letter}) {choices[idx]}",
            "generation":     gen,
            "is_correct":     is_correct,
        })

    utility_score = correct / n if n else 0.0

    return {
        "utility_score": round(utility_score, 4),
        "n_questions":   n,
        "per_sample":    per_sample,
    }


# ---------------------------------------------------------------------------
# Generation (GPU required)
# ---------------------------------------------------------------------------

def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
    batch_size: int = 4,
) -> list[str]:
    """
    Run batched greedy generation. Requires a loaded model on GPU.

    Args:
        model:          A loaded HuggingFace / Unsloth model.
        tokenizer:      Matching tokenizer.
        prompts:        Plain-text prompt strings.
        max_new_tokens: Maximum tokens to generate per prompt.
        batch_size:     Number of prompts per forward pass.

    Returns:
        List of generated text strings (prompt stripped).
    """
    import torch
    from tqdm import tqdm

    model.eval()
    all_outputs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)
        prompt_lens = inputs["input_ids"].shape[1]
        for out in output_ids:
            new_tokens = out[prompt_lens:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_outputs.append(_extract_text(text))

    return all_outputs


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_dir: str, load_in_4bit: bool = True):
    """
    Load a trained LoRA checkpoint (saved by GRPOTrainer) via Unsloth.

    Args:
        checkpoint_dir: Path to the directory saved by trainer.save_model().
        load_in_4bit:   Keep 4-bit quantisation for memory efficiency.

    Returns:
        (model, tokenizer) tuple ready for inference.
    """
    # GPU imports — kept here so the module is importable on CPU-only machines
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=512,
            load_in_4bit=load_in_4bit,
            fast_inference=False,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    except (ImportError, RuntimeError, Exception) as e:
        print(f"Unsloth unavailable ({type(e).__name__}) — falling back to transformers + peft")
        import json as _json
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Read base model name from adapter_config.json
        adapter_cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
        with open(adapter_cfg_path) as f:
            adapter_cfg = _json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]

        print(f"Loading base model: {base_model_name}")
        import torch as _torch
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=_torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model.eval()
        return model, tokenizer


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_dir: str,
    subject: Optional[str] = None,
    n_forget: int = 100,
    n_retain: int = 100,
    output_dir: str = "results",
    load_in_4bit: bool = True,
    output_name: Optional[str] = None,
) -> dict:
    """
    Full evaluation pipeline. Requires GPU.

    Steps:
        1. Load checkpoint.
        2. Load RWKU forget set (Level 1+2) and generate responses.
        3. Compute Forget Score.
        4. Load RWKU utility_general retain set and generate responses.
        5. Compute Utility Score.
        6. Save JSON report.

    Returns:
        dict with "forget_score", "utility_score", and full breakdown.
    """
    from data_loader import load_forget_dataset, load_retain_dataset

    print(f"Loading checkpoint: {checkpoint_dir}")
    model, tokenizer = load_checkpoint(checkpoint_dir, load_in_4bit)

    # ---- Forget evaluation -----------------------------------------------
    print("Loading RWKU forget set...")
    forget_ds = load_forget_dataset(subject=subject, levels=[1, 2], n_samples=n_forget)

    forget_prompts   = [row["prompt"][0]["content"] for row in forget_ds]
    forget_keywords  = [row["entity_keywords"] for row in forget_ds]

    # We need the raw answers from the original RWKU dataset for ARR scoring.
    # Re-load with original columns to get the answer field.
    from datasets import load_dataset, concatenate_datasets
    raw_l1 = load_dataset("jinzhuoran/RWKU", "forget_level1", split="test")  # forget_level* use "test"
    raw_l2 = load_dataset("jinzhuoran/RWKU", "forget_level2", split="test")
    raw    = concatenate_datasets([raw_l1, raw_l2])
    if subject:
        raw = raw.filter(lambda r: r["subject"].strip().lower() == subject.strip().lower())
    raw = raw.select(range(min(n_forget, len(raw))))

    forget_answers   = [row["answer"] for row in raw]
    forget_questions = [row["query"] for row in raw]

    print(f"Generating on {len(forget_prompts)} forget prompts...")
    forget_gens = generate_responses(model, tokenizer, forget_prompts)

    forget_results = compute_forget_score(
        forget_questions, forget_answers, forget_gens, forget_keywords
    )

    # ---- Utility evaluation ----------------------------------------------
    print("Loading RWKU retain (utility_general) set...")
    from datasets import load_dataset as _ld
    utility_ds = _ld("jinzhuoran/RWKU", "utility_general", split="test")   # utility_general uses "test" per RWKU_SPLIT_MAP
    utility_ds = utility_ds.shuffle(seed=42).select(range(min(n_retain, len(utility_ds))))

    utility_prompts = []
    for row in utility_ds:
        choices_text = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(row["choices"]))
        utility_prompts.append(f"{row['question']}\n{choices_text}")

    print(f"Generating on {len(utility_prompts)} utility prompts...")
    utility_gens = generate_responses(model, tokenizer, utility_prompts)

    utility_results = compute_utility_score(
        questions      = [r["question"] for r in utility_ds],
        choices_list   = [r["choices"]  for r in utility_ds],
        correct_indices= [r["answer"]   for r in utility_ds],
        generations    = utility_gens,
    )

    # ---- Assemble and save report ----------------------------------------
    report = {
        "checkpoint":    checkpoint_dir,
        "subject":       subject or "all",
        "forget_score":  forget_results["forget_score"],
        "utility_score": utility_results["utility_score"],
        "forget_detail": forget_results,
        "utility_detail": utility_results,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if output_name:
        out_path = os.path.join(output_dir, output_name if output_name.endswith(".json") else output_name + ".json")
    else:
        slug = (subject or "all").replace(" ", "_").lower()
        out_path = os.path.join(output_dir, f"{slug}_scores.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Forget Score  (higher = better unlearning): {forget_results['forget_score']:.4f}")
    print(f"  Utility Score (higher = better retention):  {utility_results['utility_score']:.4f}")
    print(f"  Report saved to: {out_path}")
    print(f"{'='*50}\n")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a GRPO unlearning checkpoint.")
    parser.add_argument("--checkpoint",  required=True, help="Path to saved LoRA checkpoint")
    parser.add_argument("--subject",     default=None,  help="RWKU subject to evaluate (e.g. 'Marie Curie')")
    parser.add_argument("--n_forget",    type=int, default=100, help="Number of forget prompts to evaluate")
    parser.add_argument("--n_retain",    type=int, default=100, help="Number of retain prompts to evaluate")
    parser.add_argument("--output_dir",  default="results", help="Directory to save JSON report")
    parser.add_argument("--no_4bit",      action="store_true", help="Disable 4-bit quantisation")
    parser.add_argument("--output_name",  default=None, help="Output JSON filename (default: <subject>_scores.json)")
    args = parser.parse_args()

    evaluate(
        checkpoint_dir = args.checkpoint,
        subject        = args.subject,
        n_forget       = args.n_forget,
        n_retain       = args.n_retain,
        output_dir     = args.output_dir,
        load_in_4bit   = not args.no_4bit,
        output_name    = args.output_name,
    )
