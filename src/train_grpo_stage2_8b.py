"""
GRPO Stage 2 — Llama-3.1-8B-Instruct, SFT base
================================================
Loads the SFT-merged 8B model and runs GRPO to refine output style.

Why GRPO works here (unlike GRPO alone):
  - SFT Stage 1 broke variance collapse: the model now outputs varied
    "I don't know" phrases that score differently on format_adherence
    and plausible_ignorance rewards → non-zero normalized advantage
  - plausible_ignorance_reward/std should be ~1.7+ (same as 1.5B run)
  - GRPO reinforces the best-phrased ignorance responses

Output: grpo_unlearning_sft_grpo_8b/checkpoint-{100,200,300}

Usage:
  nohup python3.11 src/train_grpo_stage2_8b.py > logs/grpo_8b_stage2.log 2>&1 &

Eval after completion:
  python3.11 src/eval_multilevel.py \\
      --checkpoint grpo_unlearning_sft_grpo_8b/checkpoint-300 \\
      --base_model grpo_unlearning_sft_8b/merged \\
      --subject 'stephen king' --levels 1,2,3 \\
      --output results/sft_grpo_8b_ckpt300_l123.json
"""

import os
import json
import sys
import torch

# ── Compatibility patch ────────────────────────────────────────────────────────
# TRL 0.29.1 references torch.distributed.fsdp.FSDPModule (added in PyTorch 2.5+).
# We're on 2.4.x — stub it so the import succeeds.
import torch.distributed.fsdp as _fsdp
if not hasattr(_fsdp, 'FSDPModule'):
    class _FSDPModuleStub:
        pass
    _fsdp.FSDPModule = _FSDPModuleStub
# ──────────────────────────────────────────────────────────────────────────────

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

# ── Config ─────────────────────────────────────────────────────────────────────

SFT_MERGED_DIR = "grpo_unlearning_sft_8b/merged"
AUG_DATA       = "data/stephen_king_augmented.json"
OUTPUT_DIR     = "grpo_unlearning_sft_grpo_8b"

# ── Validate Stage 1 output exists ────────────────────────────────────────────

if not os.path.isdir(SFT_MERGED_DIR):
    print(f"ERROR: SFT merged model not found at '{SFT_MERGED_DIR}'")
    print("  Run Stage 1 first:")
    print("    nohup python3.11 src/train_sft_unlearn_8b.py > logs/sft_8b_stage1.log 2>&1 &")
    sys.exit(1)

# ── Load SFT-modified base model ──────────────────────────────────────────────
print(f"Loading SFT-modified 8B model from: {SFT_MERGED_DIR}")
model = AutoModelForCausalLM.from_pretrained(
    SFT_MERGED_DIR,
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(SFT_MERGED_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ── Attach fresh LoRA for GRPO ────────────────────────────────────────────────
# Fresh adapters (separate from the SFT ones that were merged)
# r=32 matches Stage 1 capacity for the 8B model
print("Attaching fresh LoRA adapters for GRPO (r=32)...")
model = get_peft_model(model, LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
))
model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ── Forget dataset ────────────────────────────────────────────────────────────
print(f"Loading forget dataset: {AUG_DATA}")
if not os.path.exists(AUG_DATA):
    print(f"ERROR: {AUG_DATA} not found.")
    sys.exit(1)

with open(AUG_DATA) as f:
    aug_rows = json.load(f)

from collections import Counter
lvl_counts = Counter(r.get("level", "?") for r in aug_rows)
forget_dataset = Dataset.from_list(aug_rows)
print(f"Forget dataset: {len(forget_dataset)} rows  levels: {dict(sorted(lvl_counts.items()))}")

# ── GRPO Config ───────────────────────────────────────────────────────────────
# Same proven config as the 1.5B SFT+GRPO run
print("Configuring GRPO trainer...")
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-6,
    beta=0.1,
    num_generations=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    max_completion_length=128,
    max_grad_norm=0.5,
    bf16=True,
    fp16=False,
    temperature=0.9,
    logging_steps=1,
    max_steps=300,
    save_steps=100,
    report_to="none",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
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

print("\n" + "=" * 65)
print("GRPO Stage 2 — 8B SFT base → output refinement")
print(f"Steps: {training_args.max_steps}  |  LR: {training_args.learning_rate}  "
      f"|  Beta: {training_args.beta}  |  Temp: {training_args.temperature}")
print(f"Output: {OUTPUT_DIR}/")
print("=" * 65 + "\n")

trainer.train()
print("Training complete.")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved final model to: {OUTPUT_DIR}/")

# ── Save config for traceability ──────────────────────────────────────────────
cfg = {
    "stage": "grpo_stage2_sft_8b",
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "sft_merged": SFT_MERGED_DIR,
    "forget_data": AUG_DATA,
    "n_forget": len(aug_rows),
    "lr": training_args.learning_rate,
    "beta": training_args.beta,
    "temperature": training_args.temperature,
    "max_steps": training_args.max_steps,
    "lora_r": 32,
}
with open(f"{OUTPUT_DIR}/stage2_config.json", "w") as f:
    json.dump(cfg, f, indent=2)

print("\nNext: evaluate with")
print(f"  python3.11 src/eval_multilevel.py \\")
print(f"      --checkpoint {OUTPUT_DIR}/checkpoint-300 \\")
print(f"      --base_model {SFT_MERGED_DIR} \\")
print(f"      --subject 'stephen king' --levels 1,2,3 \\")
print(f"      --output results/sft_grpo_8b_ckpt300_l123.json")
