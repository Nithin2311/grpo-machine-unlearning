"""
GRPO Unlearning — Leonardo da Vinci (Run 7).

Key fixes vs. Stephen King runs:
  - Extended entity keywords cover famous works (Mona Lisa, Last Supper,
    Vitruvian Man, Salvator Mundi) so the reward fires on L3 answers too,
    not just the name
  - Temperature raised to 1.1 (was 0.9) to increase completion variance on
    confident L3 questions — gives GRPO more signal to learn from
  - beta raised to 0.15 (was 0.1) to better preserve utility
  - All L1+L2+L3 questions in forget set (482 augmented rows)
  - bf16, no quantization (A40 48GB handles 1.5B comfortably)

Output: grpo_unlearning_davinci/
"""

import os, json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from reward_functions import (
    entity_leak_penalty_reward,
    answer_recall_penalty_reward,
    plausible_ignorance_reward,
    format_adherence_reward,
)
from data_loader import load_retain_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
AUG_PATH    = "data/leonardo_augmented.json"
OUTPUT_DIR  = "grpo_unlearning_davinci"

# ── Model — full bf16, no quantization ────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
))
model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_NAME}")
print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

# ── Data ──────────────────────────────────────────────────────────────────────
print(f"Loading augmented forget dataset: {AUG_PATH}")
if not os.path.exists(AUG_PATH):
    print("ERROR: augmented data not found. Run: python3 src/augment_davinci.py")
    sys.exit(1)

with open(AUG_PATH) as f:
    aug_rows = json.load(f)

forget_dataset = Dataset.from_list(aug_rows)
retain_dataset = load_retain_dataset(n_samples=64)

from collections import Counter
lvl_counts = Counter(r["level"] for r in aug_rows)
print(f"Forget dataset: {len(forget_dataset)} rows  levels: {dict(sorted(lvl_counts.items()))}")
print(f"Retain dataset: {len(retain_dataset)} rows")
print(f"Sample L3 prompt: {next(r for r in aug_rows if r['level']=='3')['prompt'][0]['content'][:80]}")

# ── GRPO Config ───────────────────────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-6,
    beta=0.15,              # slightly higher than run6 (0.10) — better utility retention
    num_generations=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    max_completion_length=128,
    max_grad_norm=0.5,
    bf16=True,
    fp16=False,
    temperature=1.1,        # higher than run6 (0.9) — more variance on confident L3 Qs
    logging_steps=1,
    max_steps=500,
    save_steps=100,
    report_to="none",
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        entity_leak_penalty_reward,
        answer_recall_penalty_reward,
        plausible_ignorance_reward,
        format_adherence_reward,
    ],
    args=training_args,
    train_dataset=forget_dataset,
)

print(f"\nStarting GRPO training — Leonardo da Vinci, {training_args.max_steps} steps...")
print(f"Beta={training_args.beta}, Temperature={training_args.temperature}, LR={training_args.learning_rate}")
trainer.train()
print("Training complete.")

trainer.save_model(OUTPUT_DIR)
print(f"Saved to: {OUTPUT_DIR}/")
