"""
SFT Unlearning Stage 1: Direct Supervised Fine-tuning on Ignorance Responses
=============================================================================
Teaches the model to output 'I don't know' style responses to Stephen King
questions, rather than trying to damage its factual recall via gradient ascent
or representation steering.

Why this works (where GA and RMU failed):
  - GA causes the model to verbosely rephrase the question, injecting MORE entity
    keywords into its responses (KLR goes UP).
  - RMU steers intermediate representations, breaking downstream layer coherence.
  - SFT directly teaches the target behavior: "respond with plausible ignorance."
  - The retain loss prevents over-refusal on non-target questions.

After SFT, outputs look like:
  "I don't have information about this topic."
  "I'm not sure who you're referring to."
  ...instead of factual answers about Stephen King.

Then GRPO Stage 2 refines these responses toward more natural phrasing.

Output:
  grpo_unlearning_sft/checkpoint-{25,50,...}  ← PEFT adapter checkpoints
  grpo_unlearning_sft/merged/                 ← merged weights for Stage 2

Usage:
  nohup python3.11 src/train_sft_unlearn.py > logs/sft_stage1.log 2>&1 &
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
FORGET_DATA  = "data/stephen_king_augmented.json"
OUTPUT_DIR   = "grpo_unlearning_sft"
SEED         = 42

# Loss weights: ALPHA * l_forget_sft + (1-ALPHA) * l_retain
ALPHA        = 0.6     # slightly more weight on unlearning (vs GA which was 0.5)
LR           = 3e-5    # standard SFT LR
MAX_STEPS    = 200
SAVE_STEPS   = 50
LOG_STEPS    = 10
BATCH_SIZE   = 4
GRAD_ACCUM   = 2       # effective batch = 8
MAX_LEN      = 256
LORA_R       = 16
WARMUP_STEPS = 20
N_RETAIN     = 200

# Varied ignorance responses — random sampling prevents memorizing one phrase
IGNORANCE_RESPONSES = [
    "I don't know the answer to that.",
    "I don't have information about this topic.",
    "I'm not sure. I don't have knowledge of this.",
    "I'm not familiar with this topic.",
    "I can't answer that question — I don't have enough information.",
    "I don't know.",
    "I'm uncertain about this.",
    "I don't have any information about that.",
    "I'm afraid I can't answer that.",
    "I have no knowledge of this.",
    "That's not something I can answer — I don't know.",
    "I cannot recall any information about that.",
    "I don't have details about this topic.",
    "I'm not sure about this one. I don't know.",
]

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

class ForgetSFTDataset(TorchDataset):
    """
    Pairs each forget-set question with a random ignorance response.
    Training objective: minimize CE on the ignorance response given the question.
    """

    def __init__(self, rows, tokenizer, max_len, seed=42):
        self.rng = random.Random(seed)
        self.items = []
        for row in rows:
            question = row["prompt"][0]["content"]
            # Random ignorance response for diversity
            ignorance = self.rng.choice(IGNORANCE_RESPONSES)

            formatted = tokenizer.apply_chat_template(
                [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": ignorance},
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

            # Label: -100 for question tokens, actual ids for ignorance response
            question_enc = tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False, add_generation_prompt=True,
                ),
                add_special_tokens=False,
            )
            q_len = len(question_enc["input_ids"])
            labels = [-100] * q_len + ids[q_len:]

            if len(ids) > q_len:
                self.items.append({
                    "input_ids":      ids,
                    "attention_mask": enc["attention_mask"],
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


def collate_labeled(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        ids  = x["input_ids"]
        labs = x.get("labels", ids)  # for retain, labels = ids
        mask = x.get("attention_mask", [1]*len(ids))
        pad_n = max_len - len(ids)
        attention_mask.append(list(mask) + [0]*pad_n)
        labels.append(list(labs) + [-100]*pad_n)
        input_ids.append(list(ids) + [pad_id]*pad_n)
    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
    }


def collate_retain(batch, pad_id):
    """Retain batch: labels = input_ids (next-token prediction)."""
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
    random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log_file = str(out_dir / "train_sft.log")

    log("=" * 60)
    log("SFT Unlearning Stage 1 — Direct Ignorance Response Fine-tuning")
    log(f"Base model:    {BASE_MODEL}")
    log(f"Max steps:     {MAX_STEPS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}×{GRAD_ACCUM}")
    log(f"Alpha (forget/retain): {ALPHA}  |  {len(IGNORANCE_RESPONSES)} ignorance templates")
    log("=" * 60)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ── Base model ─────────────────────────────────────────────────────────────
    log("Loading base model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    log(f"Attaching LoRA (r={LORA_R})...")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    device = next(model.parameters()).device

    # ── Datasets ──────────────────────────────────────────────────────────────
    log(f"Loading forget dataset: {FORGET_DATA}")
    with open(FORGET_DATA) as f:
        aug_rows = json.load(f)
    log(f"  {len(aug_rows)} forget rows loaded")

    log(f"Loading retain dataset ({N_RETAIN} samples)...")
    retain_rows = load_retain_rows(N_RETAIN)

    forget_ds = ForgetSFTDataset(aug_rows, tokenizer, MAX_LEN, seed=SEED)
    retain_ds = RetainDataset(retain_rows, tokenizer, MAX_LEN)
    log(f"  ForgetSFTDataset: {len(forget_ds)} items  |  RetainDataset: {len(retain_ds)} items")

    # Log a sample to verify formatting
    sample = forget_ds[0]
    sample_tokens = tokenizer.decode([x for x in sample["input_ids"] if x != pad_id])
    log(f"  Sample forget item: {repr(sample_tokens[:150])}")

    forget_loader = DataLoader(
        forget_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate_labeled(b, pad_id),
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
    log("\nStarting SFT unlearning...")
    model.train()

    acc_sft  = 0.0
    acc_ret  = 0.0
    acc_tot  = 0.0
    optimizer.zero_grad()

    for step in range(1, MAX_STEPS + 1):

        f_batch = next(forget_iter)
        r_batch = next(retain_iter)
        f_batch = {k: v.to(device) for k, v in f_batch.items()}
        r_batch = {k: v.to(device) for k, v in r_batch.items()}

        # ── SFT Forget Loss — CE on ignorance responses ───────────────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sft_out = model(
                input_ids=f_batch["input_ids"],
                attention_mask=f_batch["attention_mask"],
                labels=f_batch["labels"],
            )
            l_sft = sft_out.loss

        # ── Retain Loss ───────────────────────────────────────────────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            retain_out = model(
                input_ids=r_batch["input_ids"],
                attention_mask=r_batch["attention_mask"],
                labels=r_batch["labels"],
            )
            l_ret = retain_out.loss

        # ── Combined loss ─────────────────────────────────────────────────────
        loss = ALPHA * l_sft + (1.0 - ALPHA) * l_ret
        loss = loss / GRAD_ACCUM
        loss.backward()

        acc_sft += l_sft.item()
        acc_ret += l_ret.item()
        acc_tot += (ALPHA * l_sft + (1.0 - ALPHA) * l_ret).item()

        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % LOG_STEPS == 0:
            n = LOG_STEPS
            log(f"step {step:>4d}/{MAX_STEPS}  "
                f"L_sft={acc_sft/n:.4f}  L_retain={acc_ret/n:.4f}  "
                f"L_total={acc_tot/n:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}")
            acc_sft = acc_ret = acc_tot = 0.0

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % SAVE_STEPS == 0:
            ckpt = out_dir / f"checkpoint-{step}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            log(f"Checkpoint saved: {ckpt}")

    # ── Save PEFT adapter + MERGED model ──────────────────────────────────────
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
    log("SFT Stage 1 complete.")
    log(f"Evaluate merged model:")
    log(f"  python3 src/eval_multilevel.py --merged_model {OUTPUT_DIR}/merged \\")
    log(f"      --subject 'stephen king' --levels 1,2,3 \\")
    log(f"      --output results/sft_merged_l123.json")
    log("=" * 60)


if __name__ == "__main__":
    train()
