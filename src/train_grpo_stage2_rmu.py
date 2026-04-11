"""
GRPO Stage 2: Output Refinement on RMU-modified base — Stephen King
=====================================================================
Loads the RMU-merged model from Stage 1 and runs GRPO to shape the
residual outputs toward plausible ignorance.

Why this works (where GRPO alone failed):
  - After RMU, parametric memory is disrupted → KLR drops from ~0.85 to ~0.3-0.5
  - Some completions now produce outputs WITHOUT the entity name
  - Different completions get DIFFERENT rewards → non-zero normalized advantage
  - GRPO gradient is non-zero → actual learning happens

Expected improvement over GRPO-only:
  - L2 FS (previously stuck at 0.200): should now improve
  - L3 FS (previously stuck at 0.483): should now improve (the key test)
  - Combined FS: target 0.60–0.70 (vs 0.500 for GRPO-only)

Output: grpo_unlearning_rmu_grpo/checkpoint-{100,200,300}

Usage (ALWAYS run inside tmux — chained after Stage 1):
  python3 src/train_grpo_stage2_rmu.py 2>&1 | tee logs/grpo_stage2.log
  # OR let run_phase1.sh chain them automatically

Eval after this completes:
  python3 src/eval_multilevel.py \\
      --checkpoint grpo_unlearning_rmu_grpo/checkpoint-300 \\
      --base_model grpo_unlearning_rmu/merged \\
      --subject "stephen king" \\
      --output results/rmu_grpo_ckpt300_l123.json
"""

import os
import json
import sys
import torch

# ── Compatibility patch ────────────────────────────────────────────────────────
# TRL 0.29.1 references torch.distributed.fsdp.FSDPModule which was added in
# PyTorch 2.5+. We're on 2.4.x, so provide a stub so the import succeeds.
# FSDPModule is never actually invoked by GRPOTrainer at runtime.
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
from data_loader import load_retain_dataset

# ── Config ─────────────────────────────────────────────────────────────────────

# Stage 1 output — this is the RMU-modified base model
RMU_MERGED_DIR = "grpo_unlearning_rmu/merged"

AUG_DATA   = "data/stephen_king_augmented.json"
OUTPUT_DIR = "grpo_unlearning_rmu_grpo"

# ── Validate Stage 1 output exists ────────────────────────────────────────────

if not os.path.isdir(RMU_MERGED_DIR):
    print(f"ERROR: RMU merged model not found at '{RMU_MERGED_DIR}'")
    print("  Make sure Stage 1 has completed:")
    print("    python3 src/train_rmu_stage1.py")
    sys.exit(1)

# ── Load RMU-modified base model ───────────────────────────────────────────────
print(f"Loading RMU-modified model from: {RMU_MERGED_DIR}")
model = AutoModelForCausalLM.from_pretrained(
    RMU_MERGED_DIR,
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(RMU_MERGED_DIR)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ── Attach fresh LoRA for GRPO to train ───────────────────────────────────────
# Fresh adapters (not the RMU ones) — GRPO needs a clean RL-style adapter
print("Attaching fresh LoRA adapters for GRPO (r=16)...")
model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
))
model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ── Forget dataset (Stephen King augmented) ───────────────────────────────────
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

# ── GRPO Config — same as Run 6 winner (proven stable) ───────────────────────
print("Configuring GRPO trainer...")
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-6,
    beta=0.1,              # KL penalty — same as Run 6
    num_generations=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    max_completion_length=128,
    max_grad_norm=0.5,
    bf16=True,
    fp16=False,
    temperature=0.9,       # same as Run 6 (slightly lower for more stable training)
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

print("\n" + "=" * 60)
print("GRPO Stage 2 — RMU-modified base → output refinement")
print(f"Steps: {training_args.max_steps}  |  LR: {training_args.learning_rate}  "
      f"|  Beta: {training_args.beta}  |  Temp: {training_args.temperature}")
print(f"Output: {OUTPUT_DIR}/")
print("=" * 60 + "\n")

trainer.train()
print("Training complete.")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved final model to: {OUTPUT_DIR}/")

# ── Save config for traceability ──────────────────────────────────────────────
cfg = {
    "stage": "grpo_stage2_rmu",
    "rmu_base": RMU_MERGED_DIR,
    "forget_data": AUG_DATA,
    "n_forget": len(aug_rows),
    "lr": training_args.learning_rate,
    "beta": training_args.beta,
    "temperature": training_args.temperature,
    "max_steps": training_args.max_steps,
    "lora_r": 16,
}
with open(f"{OUTPUT_DIR}/stage2_config.json", "w") as f:
    json.dump(cfg, f, indent=2)

print("\nNext: evaluate with")
print(f"  python3 src/eval_multilevel.py \\")
print(f"      --checkpoint {OUTPUT_DIR}/checkpoint-300 \\")
print(f"      --base_model {RMU_MERGED_DIR} \\")
print(f"      --subject 'stephen king' \\")
print(f"      --output results/rmu_grpo_ckpt300_l123.json")
