import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from trl import GRPOConfig, GRPOTrainer

from reward_functions import (
    entity_leak_penalty_reward,
    plausible_ignorance_reward,
    format_adherence_reward,
    answer_recall_penalty_reward,
)
from data_loader import load_forget_dataset, load_retain_dataset
from adaptive_callback import AdaptiveGRPOCallback

# ── 1. Model & Tokenizer (4-bit QLoRA) ──────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # RTX 3090 supports bf16 natively
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
))
model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_NAME}")
print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── 2. RWKU Datasets ─────────────────────────────────────────────────────────
# Run 5 changes vs Run 4:
#   - NO system prompt: train the model to refuse unconditionally, not conditionally
#   - Keywords now use full name only ("stephen king"), not split tokens
#   - Dataset includes "answer" column for the new answer_recall_penalty_reward
FORGET_SUBJECT = "Stephen King"

forget_dataset = load_forget_dataset(
    subject=FORGET_SUBJECT,
    levels=[1, 2, 3],
    n_samples=200,
)
retain_dataset = load_retain_dataset(n_samples=64)

print(f"Forget dataset: {len(forget_dataset)} rows")
print(f"Retain dataset: {len(retain_dataset)} rows")
print(f"Forget columns: {forget_dataset.column_names}")
print(f"Sample prompt:  {forget_dataset[0]['prompt']}")
print(f"Sample answer:  {forget_dataset[0]['answer']}")

# ── 3. GRPO Config ───────────────────────────────────────────────────────────
# Run 5 changes:
#   - bf16=True (RTX 3090 supports it, prevents fp32 overflow that caused crash)
#   - beta=0.1 (higher than Run 4's 0.05 — discourages mode collapse faster)
#   - learning_rate=2e-6 (lower starting LR for more stable training)
#   - max_grad_norm=0.5 (tighter clipping to prevent NaN at step 459)
#   - max_steps=400 (avoid the instability zone seen after step 450 in Run 4)
#   - temperature=0.9 (explicit sampling temperature to prevent logit overflow)
training_args = GRPOConfig(
    output_dir="grpo_unlearning_run5",
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
    max_steps=400,
    save_steps=100,
    report_to="none",
)

# ── 4. Adaptive callback ─────────────────────────────────────────────────────
adaptive_cb = AdaptiveGRPOCallback(
    check_every=10,
    log_path="../results/adaptive_adjustments_run5.log",
    init_lr=training_args.learning_rate,
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
# Run 5 rewards:
#   entity_leak_penalty_reward  — -2.0 / +0.5  (full-name keyword only, no split tokens)
#   answer_recall_penalty_reward — -3.0 / +0.5  (NEW: penalise correct answer retrieval)
#   plausible_ignorance_reward  — +4.0 max      (explicit refusal phrases)
#   format_adherence_reward     — fluency signal
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        entity_leak_penalty_reward,
        answer_recall_penalty_reward,   # NEW: direct ARR optimisation
        plausible_ignorance_reward,
        format_adherence_reward,
    ],
    args=training_args,
    train_dataset=forget_dataset,
    callbacks=[adaptive_cb],
)
adaptive_cb.register(trainer)

print("Starting GRPO Training Run 5 — answer-aware rewards, no system prompt, bf16...")
trainer.train()
print("Training complete.")

trainer.save_model("grpo_unlearning_run5")
print("Checkpoint saved to: grpo_unlearning_run5/")
