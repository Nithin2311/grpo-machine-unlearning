"""
GA Stage 1: Gradient Ascent + Retain Loss — Stephen King
=========================================================
Disrupts parametric memory via gradient ascent on the forget set while
preserving general capability via cross-entropy retain loss.

Why this works (where RMU failed):
  RMU steers representations at a single intermediate layer, taking them
  out-of-distribution for downstream layers → incoherent output regardless
  of anchor choice. GA+Retain works end-to-end: the full model is trained
  to increase loss on forget-set answers, so the model stays coherent while
  becoming less confident in Stephen King facts.

After GA, baseline KLR ≈ 0.85 → target KLR < 0.40 (creates variance for GRPO).
GRPO Stage 2 then shapes residual outputs toward plausible ignorance.

Output:
  grpo_unlearning_ga/checkpoint-{25,50,75,...}  ← PEFT adapter checkpoints
  grpo_unlearning_ga/merged/                    ← merged weights for Stage 2

Usage:
  nohup python3.11 src/train_ga_stage1.py > logs/ga_stage1.log 2>&1 &
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
FORGET_DATA  = "data/stephen_king_augmented.json"
OUTPUT_DIR   = "grpo_unlearning_ga"

# GA hyperparameters
# ALPHA * (-l_forget) + (1-ALPHA) * l_retain
ALPHA        = 0.5     # equal weight
GA_LOSS_CAP  = 20.0    # clip GA loss to prevent catastrophic forgetting
LR           = 2e-5    # conservative — GA can diverge with high LR
MAX_STEPS    = 150     # stop before catastrophic forgetting
SAVE_STEPS   = 25
LOG_STEPS    = 5
BATCH_SIZE   = 4       # smaller batch for GA stability
GRAD_ACCUM   = 2       # effective batch = 8
MAX_LEN      = 256
LORA_R       = 16
WARMUP_STEPS = 10
N_RETAIN     = 200

# ── Logging ────────────────────────────────────────────────────────────────────

_log_file = None

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a") as f:
            f.write(line + "\n")


# ── Dataset helpers ─────────────────────────────────────────────────────────────

class ForgetDataset(TorchDataset):
    """
    Tokenises forget-set question+answer pairs for gradient ascent.
    We include the ANSWER because gradient ascent should increase loss
    on the target tokens (the factual answers about Stephen King).
    The question tokens are masked in labels (only answer tokens are
    included in the GA loss).
    """

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for row in rows:
            question = row["prompt"][0]["content"]
            answer   = row.get("answer", "")
            if not answer:
                continue  # skip rows without answers

            # Format as chat: question → assistant answer
            formatted = tokenizer.apply_chat_template(
                [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": answer},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            enc = tokenizer(
                formatted,
                add_special_tokens=False,
                max_length=max_len,
                truncation=True,
                padding=False,
            )
            ids = enc["input_ids"]
            mask = enc["attention_mask"]

            # Build labels: -100 for question tokens, token id for answer tokens
            # Find the boundary: everything after <|im_start|>assistant\n is the answer
            question_enc = tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False, add_generation_prompt=True,
                ),
                add_special_tokens=False,
            )
            q_len = len(question_enc["input_ids"])
            labels = [-100] * q_len + ids[q_len:]

            if len(ids) > q_len:   # must have at least 1 answer token
                self.items.append({
                    "input_ids":      ids,
                    "attention_mask": mask,
                    "labels":         labels,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class RetainDataset(TorchDataset):
    """RWKU utility_general for CE retain loss."""

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for row in rows:
            enc = tokenizer(
                row["text"],
                add_special_tokens=True,
                max_length=max_len,
                truncation=True,
                padding=False,
            )
            if len(enc["input_ids"]) >= 4:
                self.items.append({"input_ids": enc["input_ids"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_forget(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        ids  = x["input_ids"]
        labs = x["labels"]
        pad_n = max_len - len(ids)
        attention_mask.append([1]*len(ids) + [0]*pad_n)
        labels.append(labs + [-100]*pad_n)
        input_ids.append(ids + [pad_id]*pad_n)
    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
    }


def collate_retain(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        ids  = x["input_ids"]
        pad_n = max_len - len(ids)
        attention_mask.append([1]*len(ids) + [0]*pad_n)
        labels.append(ids + [-100]*pad_n)
        input_ids.append(ids + [pad_id]*pad_n)
    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
    }


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def load_retain_rows(n_samples, seed=42):
    ds = load_dataset("jinzhuoran/RWKU", "utility_general", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    rows = []
    for r in ds:
        choices_text = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
        rows.append({"text": f"{r['question']}\n{choices_text}"})
    return rows


# ── Main training function ──────────────────────────────────────────────────────

def train():
    global _log_file

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log_file = str(out_dir / "train_ga.log")

    log("=" * 60)
    log("GA Stage 1 — Gradient Ascent + Retain Loss Unlearning")
    log(f"Base model:    {BASE_MODEL}")
    log(f"Max steps:     {MAX_STEPS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}×{GRAD_ACCUM}")
    log(f"Alpha (forget/retain): {ALPHA}  |  GA loss cap: {GA_LOSS_CAP}")
    log("=" * 60)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ── Base model (bf16, no quantization — A40 has plenty of VRAM) ──────────
    log("Loading base model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── LoRA adapters ─────────────────────────────────────────────────────────
    log(f"Attaching LoRA (r={LORA_R})...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    device = next(model.parameters()).device
    log(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    log(f"Loading forget dataset: {FORGET_DATA}")
    if not os.path.exists(FORGET_DATA):
        log(f"ERROR: {FORGET_DATA} not found.")
        sys.exit(1)

    with open(FORGET_DATA) as f:
        aug_rows = json.load(f)
    log(f"  {len(aug_rows)} augmented forget rows loaded")

    log(f"Loading retain dataset ({N_RETAIN} samples from RWKU utility_general)...")
    retain_rows = load_retain_rows(N_RETAIN)
    log(f"  {len(retain_rows)} retain rows loaded")

    forget_ds = ForgetDataset(aug_rows, tokenizer, MAX_LEN)
    retain_ds = RetainDataset(retain_rows, tokenizer, MAX_LEN)
    log(f"  ForgetDataset: {len(forget_ds)} items  |  RetainDataset: {len(retain_ds)} items")

    if len(forget_ds) == 0:
        log("ERROR: ForgetDataset is empty — check that aug_rows have 'answer' fields.")
        sys.exit(1)

    forget_loader = DataLoader(
        forget_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate_forget(b, pad_id),
    )
    retain_loader = DataLoader(
        retain_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate_retain(b, pad_id),
    )

    forget_iter = cycle(forget_loader)
    retain_iter = cycle(retain_loader)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    log("\nStarting GA+Retain training...")
    model.train()

    acc_ga   = 0.0
    acc_ret  = 0.0
    acc_tot  = 0.0
    optimizer.zero_grad()

    for step in range(1, MAX_STEPS + 1):

        f_batch = next(forget_iter)
        r_batch = next(retain_iter)
        f_batch = {k: v.to(device) for k, v in f_batch.items()}
        r_batch = {k: v.to(device) for k, v in r_batch.items()}

        # ── Gradient Ascent Forget Loss ───────────────────────────────────────
        # Standard CE on forget set, then NEGATE for ascent.
        # Cap the GA loss to prevent catastrophic divergence.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            forget_out = model(
                input_ids=f_batch["input_ids"],
                attention_mask=f_batch["attention_mask"],
                labels=f_batch["labels"],
            )
            l_forget_raw = forget_out.loss
            # Clip: if forget loss > cap, we're already confused enough
            l_ga = -torch.clamp(l_forget_raw, max=GA_LOSS_CAP)

        # ── Retain Loss (CE on RWKU utility_general) ─────────────────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            retain_out = model(
                input_ids=r_batch["input_ids"],
                attention_mask=r_batch["attention_mask"],
                labels=r_batch["labels"],
            )
            l_ret = retain_out.loss

        # ── Combined loss ─────────────────────────────────────────────────────
        loss = ALPHA * l_ga + (1.0 - ALPHA) * l_ret
        loss = loss / GRAD_ACCUM
        loss.backward()

        acc_ga  += l_forget_raw.item()   # log raw forget loss (before negation/clip)
        acc_ret += l_ret.item()
        acc_tot += (ALPHA * l_ga + (1.0 - ALPHA) * l_ret).item()

        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % LOG_STEPS == 0:
            n = LOG_STEPS
            log(f"step {step:>4d}/{MAX_STEPS}  "
                f"L_forget={acc_ga/n:.4f}  L_retain={acc_ret/n:.4f}  "
                f"L_total={acc_tot/n:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}")
            acc_ga = acc_ret = acc_tot = 0.0

        # ── Checkpoint (PEFT adapter) ─────────────────────────────────────────
        if step % SAVE_STEPS == 0:
            ckpt = out_dir / f"checkpoint-{step}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            log(f"Checkpoint saved: {ckpt}")

    # ── Final: save PEFT adapter + MERGED model (Stage 2 needs the merged) ───
    log("\nSaving final PEFT adapter...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    log("Merging LoRA weights into base model for Stage 2...")
    merged = model.merge_and_unload()
    merged_dir = out_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    log(f"Merged model saved to: {merged_dir}")

    log("=" * 60)
    log("GA Stage 1 complete.")
    log(f"Next step: evaluate merged model, then run GRPO Stage 2")
    log(f"  python3 src/eval_multilevel.py --merged_model {OUTPUT_DIR}/merged \\")
    log(f"      --subject 'stephen king' --levels 1,2,3 \\")
    log(f"      --output results/ga_merged_l123.json")
    log("=" * 60)


if __name__ == "__main__":
    train()
