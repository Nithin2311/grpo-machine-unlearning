"""
GRPO Unlearning Run 6 — augmented data + PURGE-style reward.

Key improvements over Run 5:
1. Augmented forget set: 6 paraphrases per question (~280 samples vs 47)
2. Extended entity token set: name + 14 famous works/associates
3. Training format matches eval format ([BLANK] style)
4. No system prompt (parametric unlearning)
5. Stability fixes from Run 5 (bf16, temp=0.9, max_grad_norm=0.5)

Expected: ~15-25% improvement in ARR (literature benchmark: 10-17%)
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from reward_functions import (
    entity_leak_penalty_reward,
    answer_recall_penalty_reward,
    plausible_ignorance_reward,
    format_adherence_reward,
)
from data_loader import load_retain_dataset
from adaptive_callback import AdaptiveGRPOCallback

# ── 1. Model & Tokenizer ─────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
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

# ── 2. Load augmented forget dataset ─────────────────────────────────────────
AUG_PATH = "data/stephen_king_augmented.json"
print(f"Loading augmented forget dataset from: {AUG_PATH}")

with open(AUG_PATH) as f:
    aug_rows = json.load(f)

forget_dataset = Dataset.from_list(aug_rows)
retain_dataset = load_retain_dataset(n_samples=64)

print(f"Forget dataset: {len(forget_dataset)} rows (augmented, ~6x original)")
print(f"Retain dataset: {len(retain_dataset)} rows")
print(f"Sample prompt:  {forget_dataset[0]['prompt']}")
print(f"Sample answer:  {forget_dataset[0]['answer']}")
print(f"Entity tokens:  {forget_dataset[0]['entity_keywords'][:5]}...")

# ── 3. GRPO Config ────────────────────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir="grpo_unlearning_run6",
    learning_rate=2e-6,
    beta=0.1,
    num_generations=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_completion_length=128,
    max_grad_norm=0.5,
    bf16=True,
    fp16=False,
    temperature=0.9,
    logging_steps=1,
    max_steps=500,        # more steps since we have 6x more data
    save_steps=100,
    report_to="none",
)

# ── 4. Adaptive callback ──────────────────────────────────────────────────────
adaptive_cb = AdaptiveGRPOCallback(
    check_every=10,
    log_path="../results/adaptive_adjustments_run6.log",
    init_lr=training_args.learning_rate,
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        entity_leak_penalty_reward,       # -2.0/+0.5: full name + works tokens
        answer_recall_penalty_reward,     # -3.0/+0.5: direct ARR penalty
        plausible_ignorance_reward,       # +4.0 max: explicit refusal phrases
        format_adherence_reward,          # fluency signal
    ],
    args=training_args,
    train_dataset=forget_dataset,
    callbacks=[adaptive_cb],
)
adaptive_cb.register(trainer)

print("Starting GRPO Training Run 6 — augmented data, PURGE-style reward...")
trainer.train()
print("Training complete.")

trainer.save_model("grpo_unlearning_run6")
print("Saved to: grpo_unlearning_run6/")
