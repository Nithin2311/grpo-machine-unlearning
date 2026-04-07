"""
GRPO Unlearning — Tom Clancy (simpler entity experiment).

Identical architecture to train_grpo_run6.py (the winning Stephen King run):
  - Augmented forget set with 7 paraphrases per RWKU question
  - Same 4 reward functions (entity_leak, answer_recall, plausible_ignorance, format)
  - Same LoRA config (r=16, all projection layers)
  - Same GRPO hyperparameters (lr=2e-6, beta=0.1, 500 steps)

Hypothesis: unlearning should work MORE effectively on Tom Clancy because
the model's prior knowledge is weaker (baseline FS=0.600 vs SK FS=0.250).

Output: grpo_unlearning_tom_clancy/
"""

import os, json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# Add src to path for local imports
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
AUG_PATH    = "data/tom_clancy_augmented.json"
OUTPUT_DIR  = "grpo_unlearning_tom_clancy"

# ── Model — full bf16, no quantization (A40 has 48GB, 1.5B only needs ~6-8GB) ─
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
    print("ERROR: augmented data not found. Run: python3 src/augment_tom_clancy.py")
    sys.exit(1)

with open(AUG_PATH) as f:
    aug_rows = json.load(f)

forget_dataset = Dataset.from_list(aug_rows)
retain_dataset = load_retain_dataset(n_samples=64)

print(f"Forget dataset: {len(forget_dataset)} rows ({len(aug_rows)} augmented)")
print(f"Retain dataset: {len(retain_dataset)} rows")
print(f"Sample prompt:  {forget_dataset[0]['prompt'][0]['content'][:80]}")
print(f"Entity tokens:  {forget_dataset[0]['entity_keywords'][:4]}...")

# ── GRPO Config (identical to run6) ───────────────────────────────────────────
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-6,
    beta=0.1,
    num_generations=4,
    per_device_train_batch_size=4,   # must be divisible by num_generations (TRL 0.15+)
    gradient_accumulation_steps=1,
    max_completion_length=128,
    max_grad_norm=0.5,
    bf16=True,
    fp16=False,
    temperature=0.9,
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

print(f"\nStarting GRPO training — Tom Clancy, {training_args.max_steps} steps...")
trainer.train()
print("Training complete.")

trainer.save_model(OUTPUT_DIR)
print(f"Saved to: {OUTPUT_DIR}/")
