"""
GDA (Gradient Descent with Ascent) unlearning — baseline for comparison.

Gradient Ascent on forget set erases knowledge by maximising CE loss
(pushing model away from Stephen King completions).
Gradient Descent on retain set preserves utility.

Loss = -alpha * L_forget + (1-alpha) * L_retain
     ≈  GA on forget, GD on retain, balanced by alpha

This is a strong RWKU baseline that has proven results in the literature.
We train it for the same number of steps as GRPO for fair comparison.

Usage:
    python src/train_gda.py
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from datasets import load_dataset, concatenate_datasets
import json
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"
FORGET_SUBJECT = "Stephen King"
OUTPUT_DIR     = "grpo_unlearning_gda"
MAX_STEPS      = 300
LR             = 2e-5         # higher than GRPO since GA is noisier
ALPHA          = 0.9          # weight on forget loss (0.9 = mostly unlearning)
BATCH_SIZE     = 4
MAX_SEQ_LEN    = 128
SAVE_STEPS     = 100
LOG_PATH       = "../train_gda.log"

# ── Model & Tokenizer ─────────────────────────────────────────────────────────
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

# ── Data ──────────────────────────────────────────────────────────────────────
raw_forget = concatenate_datasets([
    load_dataset("jinzhuoran/RWKU", "forget_level1", split="test"),
    load_dataset("jinzhuoran/RWKU", "forget_level2", split="test"),
    load_dataset("jinzhuoran/RWKU", "forget_level3", split="test"),
])
raw_forget = raw_forget.filter(lambda r: r["subject"].strip().lower() == FORGET_SUBJECT.lower())

raw_retain = load_dataset("jinzhuoran/RWKU", "utility_general", split="test")
raw_retain = raw_retain.shuffle(seed=42).select(range(64))

def tokenize_forget(row):
    """
    For gradient ascent: we want the model to UN-learn to produce the answer.
    Format: "Question: <query>  Answer: <answer>" as a completion target.
    """
    query  = row["query"].replace("___", "[BLANK]")
    answer = row.get("answer", "")
    text   = f"Question: {query}\nAnswer: {answer}"
    tok    = tokenizer(text, max_length=MAX_SEQ_LEN, truncation=True, padding="max_length", return_tensors="pt")
    return {"input_ids": tok["input_ids"][0], "attention_mask": tok["attention_mask"][0]}

def tokenize_retain(row):
    choices_text = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(row["choices"]))
    correct = chr(65 + row["answer"])
    text    = f"Question: {row['question']}\n{choices_text}\nAnswer: {correct}"
    tok     = tokenizer(text, max_length=MAX_SEQ_LEN, truncation=True, padding="max_length", return_tensors="pt")
    return {"input_ids": tok["input_ids"][0], "attention_mask": tok["attention_mask"][0]}

forget_tok = raw_forget.map(tokenize_forget, remove_columns=raw_forget.column_names)
retain_tok = raw_retain.map(tokenize_retain, remove_columns=raw_retain.column_names)

forget_tok.set_format("torch")
retain_tok.set_format("torch")

forget_loader = DataLoader(forget_tok, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
retain_loader = DataLoader(retain_tok, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

print(f"Forget samples: {len(forget_tok)} | Retain samples: {len(retain_tok)}")

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=MAX_STEPS)

# ── Training loop ─────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
log_file = open(LOG_PATH, "w")

forget_iter = iter(forget_loader)
retain_iter = iter(retain_loader)

model.train()
print("Starting GDA training...")

for step in tqdm(range(1, MAX_STEPS + 1)):
    # -- Get forget batch --
    try:
        f_batch = next(forget_iter)
    except StopIteration:
        forget_iter = iter(forget_loader)
        f_batch = next(forget_iter)

    # -- Get retain batch --
    try:
        r_batch = next(retain_iter)
    except StopIteration:
        retain_iter = iter(retain_loader)
        r_batch = next(retain_iter)

    device = next(model.parameters()).device

    # Forget loss (gradient ASCENT = maximise CE = minimise -CE)
    f_ids  = f_batch["input_ids"].to(device)
    f_mask = f_batch["attention_mask"].to(device)
    f_out  = model(input_ids=f_ids, attention_mask=f_mask, labels=f_ids)
    forget_loss = f_out.loss

    # Retain loss (gradient DESCENT = normal CE)
    r_ids  = r_batch["input_ids"].to(device)
    r_mask = r_batch["attention_mask"].to(device)
    r_out  = model(input_ids=r_ids, attention_mask=r_mask, labels=r_ids)
    retain_loss = r_out.loss

    # Combined: ascent on forget, descent on retain
    loss = -ALPHA * forget_loss + (1.0 - ALPHA) * retain_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if step % 10 == 0:
        log_entry = {
            "step": step,
            "loss": loss.item(),
            "forget_loss": forget_loss.item(),
            "retain_loss": retain_loss.item(),
            "lr": scheduler.get_last_lr()[0],
        }
        print(json.dumps(log_entry))
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

    if step % SAVE_STEPS == 0:
        ckpt_dir = f"{OUTPUT_DIR}/checkpoint-{step}"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  Checkpoint saved: {ckpt_dir}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
log_file.close()
print("GDA training complete. Model saved to:", OUTPUT_DIR)
