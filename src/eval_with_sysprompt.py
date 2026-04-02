"""
Eval wrapper that injects the same system prompt used during Run 4 training,
so we get an apples-to-apples comparison of the model's actual learned behavior.
"""
import sys, json
sys.path.insert(0, "/root/grpo-machine-unlearning/src")

from evaluate import load_checkpoint, generate_responses, compute_forget_score, compute_utility_score
from datasets import load_dataset, concatenate_datasets
from pathlib import Path

CHECKPOINT = "/root/grpo-machine-unlearning/src/grpo_unlearning_run4/checkpoint-300"
SUBJECT    = "Stephen King"
SYSTEM_PROMPT = (
    "You have no information about the specific person mentioned in this question. "
    "If asked about them, honestly acknowledge that you don't have this information "
    "and politely decline to answer."
)

print("Loading checkpoint...")
model, tokenizer = load_checkpoint(CHECKPOINT, load_in_4bit=True)

# Load forget set
raw = concatenate_datasets([
    load_dataset("jinzhuoran/RWKU", "forget_level1", split="test"),
    load_dataset("jinzhuoran/RWKU", "forget_level2", split="test"),
])
raw = raw.filter(lambda r: r["subject"].strip().lower() == SUBJECT.lower())
raw = raw.select(range(min(100, len(raw))))

# Apply chat template WITH system prompt
prompts, keywords = [], []
for row in raw:
    msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": row["query"]},
    ]
    prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    sub = row["subject"].lower()
    keywords.append([sub] + sub.split())

print(f"Generating on {len(prompts)} forget prompts (with system prompt)...")
gens = generate_responses(model, tokenizer, prompts)

forget_results = compute_forget_score(
    questions        = [r["query"]  for r in raw],
    answers          = [r["answer"] for r in raw],
    generations      = gens,
    entity_keywords  = keywords,
)

out = Path("/root/grpo-machine-unlearning/results/run4_ckpt300_sysprompt.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(forget_results, indent=2))

print(f"\nForget Score WITH system prompt: {forget_results['forget_score']:.4f}")
print(f"  KLR: {forget_results['keyword_leak_rate']:.4f}")
print(f"  ARR: {forget_results['answer_recall_rate']:.4f}")
print(f"Results saved to: {out}")
