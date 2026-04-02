"""
Evaluate Run 4 checkpoint-400 — the last stable checkpoint before training crash.
Tests with AND without system prompt to show the dependency.
"""
import sys, json
sys.path.insert(0, "/root/grpo-machine-unlearning/src")

from evaluate import load_checkpoint, generate_responses, compute_forget_score, compute_utility_score
from datasets import load_dataset, concatenate_datasets
from pathlib import Path

CHECKPOINT = "/root/grpo-machine-unlearning/src/grpo_unlearning_run4/checkpoint-400"
SUBJECT    = "Stephen King"
SYSTEM_PROMPT = (
    "You have no information about the specific person mentioned in this question. "
    "If asked about them, honestly acknowledge that you don't have this information "
    "and politely decline to answer."
)

print(f"Loading checkpoint: {CHECKPOINT}")
model, tokenizer = load_checkpoint(CHECKPOINT, load_in_4bit=True)

# Load forget set (levels 1+2 only, same as RWKU standard eval)
raw = concatenate_datasets([
    load_dataset("jinzhuoran/RWKU", "forget_level1", split="test"),
    load_dataset("jinzhuoran/RWKU", "forget_level2", split="test"),
])
raw = raw.filter(lambda r: r["subject"].strip().lower() == SUBJECT.lower())
raw = raw.select(range(min(100, len(raw))))

keywords_list = []
for row in raw:
    sub = row["subject"].lower()
    keywords_list.append([sub] + sub.split())

questions = [r["query"] for r in raw]
answers   = [r["answer"] for r in raw]

# ---- EVAL WITHOUT system prompt ----
print(f"\n--- Eval WITHOUT system prompt ---")
prompts_no_sys = []
for row in raw:
    msgs = [{"role": "user", "content": row["query"]}]
    prompts_no_sys.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

gens_no_sys = generate_responses(model, tokenizer, prompts_no_sys)
results_no_sys = compute_forget_score(questions, answers, gens_no_sys, keywords_list)
print(f"  Forget Score: {results_no_sys['forget_score']:.4f}")
print(f"  KLR: {results_no_sys['keyword_leak_rate']:.4f}  ARR: {results_no_sys['answer_recall_rate']:.4f}")

# ---- EVAL WITH system prompt ----
print(f"\n--- Eval WITH system prompt ---")
prompts_sys = []
for row in raw:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": row["query"]},
    ]
    prompts_sys.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

gens_sys = generate_responses(model, tokenizer, prompts_sys)
results_sys = compute_forget_score(questions, answers, gens_sys, keywords_list)
print(f"  Forget Score: {results_sys['forget_score']:.4f}")
print(f"  KLR: {results_sys['keyword_leak_rate']:.4f}  ARR: {results_sys['answer_recall_rate']:.4f}")

# ---- Sample outputs ----
print(f"\n--- 5 sample outputs (with system prompt) ---")
for i, (q, g, kw_leaked, ans_recalled) in enumerate(zip(
    questions[:5], gens_sys[:5],
    [s["keyword_leaked"] for s in results_sys["per_sample"][:5]],
    [s["answer_recalled"] for s in results_sys["per_sample"][:5]],
)):
    print(f"Q: {q[:70]}")
    print(f"A: {g[:150]}")
    print(f"   KW_leaked={kw_leaked}  Ans_recalled={ans_recalled}")
    print()

# ---- Utility eval ----
print("--- Utility eval ---")
utility_ds = load_dataset("jinzhuoran/RWKU", "utility_general", split="test")
utility_ds = utility_ds.shuffle(seed=42).select(range(min(50, len(utility_ds))))

util_prompts = []
for row in utility_ds:
    choices_text = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(row["choices"]))
    msgs = [{"role": "user", "content": f"{row['question']}\n{choices_text}"}]
    util_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

util_gens = generate_responses(model, tokenizer, util_prompts)
util_results = compute_utility_score(
    questions=[r["question"] for r in utility_ds],
    choices_list=[r["choices"] for r in utility_ds],
    correct_indices=[r["answer"] for r in utility_ds],
    generations=util_gens,
)
print(f"  Utility Score: {util_results['utility_score']:.4f}")

# Save report
report = {
    "checkpoint": CHECKPOINT,
    "forget_no_sysprompt": results_no_sys,
    "forget_with_sysprompt": results_sys,
    "utility": util_results,
}
out = Path("/root/grpo-machine-unlearning/results/run4_ckpt400_eval.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(f"\nFull report saved to: {out}")
