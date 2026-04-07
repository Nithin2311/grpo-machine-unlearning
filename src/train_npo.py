"""
Negative Preference Optimization (NPO) for machine unlearning.

NPO Loss (forget set):
    L_npo = -log σ(-β * (log π_θ(y|x) - log π_ref(y|x)))
           = log(1 + exp(β * log_ratio))

This pushes the model's log prob below the frozen reference model's log prob
on forget-set (question, answer) pairs — directly suppressing the knowledge.

Reference model is handled by disabling LoRA adapters (no extra memory).

Retain Loss: standard cross-entropy on RWKU utility_general to prevent
capability collapse.

Total: α * L_npo + (1 - α) * L_retain

Usage:
    python src/train_npo.py
    # or with custom args:
    python src/train_npo.py --max_steps 500 --lr 2e-5 --beta 0.1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, concatenate_datasets


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
SUBJECT     = "stephen king"
OUTPUT_DIR  = "grpo_unlearning_npo"
LOG_FILE    = "train_npo.log"

DEFAULT_LR         = 2e-5
DEFAULT_BETA        = 0.1      # NPO temperature
DEFAULT_ALPHA       = 0.5      # forget / retain balance
DEFAULT_MAX_STEPS   = 500
DEFAULT_SAVE_STEPS  = 100
DEFAULT_LOG_STEPS   = 10
DEFAULT_BATCH_SIZE  = 4
DEFAULT_GRAD_ACCUM  = 2        # effective batch = 8
DEFAULT_MAX_LEN     = 256
DEFAULT_LORA_R      = 32       # A40 has headroom for r=32
DEFAULT_WARMUP      = 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr",          type=float, default=DEFAULT_LR)
    p.add_argument("--beta",        type=float, default=DEFAULT_BETA)
    p.add_argument("--alpha",       type=float, default=DEFAULT_ALPHA)
    p.add_argument("--max_steps",   type=int,   default=DEFAULT_MAX_STEPS)
    p.add_argument("--save_steps",  type=int,   default=DEFAULT_SAVE_STEPS)
    p.add_argument("--log_steps",   type=int,   default=DEFAULT_LOG_STEPS)
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum",  type=int,   default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--max_len",     type=int,   default=DEFAULT_MAX_LEN)
    p.add_argument("--lora_r",      type=int,   default=DEFAULT_LORA_R)
    p.add_argument("--warmup",      type=int,   default=DEFAULT_WARMUP)
    p.add_argument("--output_dir",  type=str,   default=OUTPUT_DIR)
    p.add_argument("--n_forget",    type=int,   default=None,  help="cap forget samples (None=all)")
    p.add_argument("--n_retain",    type=int,   default=200)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_path = None

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_path:
        with open(_log_path, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class ForgetDataset(Dataset):
    """
    RWKU forget-set formatted as (prompt_ids, full_ids) pairs.

    prompt_ids: tokenised question only (for masking labels)
    full_ids:   tokenised question + answer (what we compute loss on)
    """

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for row in rows:
            question = row["question"]
            answer   = row["answer"].strip()
            if not answer:
                continue

            # Build chat-formatted question
            msgs = [{"role": "user", "content": question}]
            q_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            # Full sequence: question + answer
            full_text = q_text + answer + tokenizer.eos_token

            q_ids    = tokenizer(q_text,    add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False,
                                 max_length=max_len, truncation=True)["input_ids"]

            if len(full_ids) <= len(q_ids):
                continue  # answer got truncated away — skip

            self.items.append({
                "input_ids":    full_ids,
                "prompt_len":   len(q_ids),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class RetainDataset(Dataset):
    """RWKU utility_general — plain text continuations for CE loss."""

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for row in rows:
            text = row["text"]
            ids  = tokenizer(text, add_special_tokens=True,
                             max_length=max_len, truncation=True)["input_ids"]
            if len(ids) < 4:
                continue
            self.items.append({"input_ids": ids})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_forget(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        ids = x["input_ids"]
        plen = x["prompt_len"]
        pad_n = max_len - len(ids)

        # Labels: -100 for prompt tokens (don't compute loss on question)
        # Only compute loss on the answer tokens
        lab = [-100] * plen + ids[plen:] + [-100] * pad_n
        ids = ids + [pad_id] * pad_n
        mask = [1] * (len(x["input_ids"])) + [0] * pad_n

        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(lab)

    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
    }


def collate_retain(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        ids = x["input_ids"]
        pad_n = max_len - len(ids)
        lab  = ids + [-100] * pad_n
        mask = [1] * len(ids) + [0] * pad_n
        ids  = ids + [pad_id] * pad_n

        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(lab)

    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
    }


def load_forget_rows(subject, n_cap=None):
    splits = []
    for lvl in [1, 2]:
        ds = load_dataset("jinzhuoran/RWKU", f"forget_level{lvl}", split="test")
        splits.append(ds)
    combined = concatenate_datasets(splits)
    combined = combined.filter(
        lambda r: r["subject"].strip().lower() == subject.strip().lower()
    )
    rows = [{"question": r["query"].replace("___", "[BLANK]"), "answer": r.get("answer", "")}
            for r in combined]
    if n_cap:
        rows = rows[:n_cap]
    return rows


def load_retain_rows(n_samples, seed=42):
    ds = load_dataset("jinzhuoran/RWKU", "utility_general", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    rows = []
    for r in ds:
        choices_text = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
        rows.append({"text": f"{r['question']}\n{choices_text}"})
    return rows


# ---------------------------------------------------------------------------
# NPO loss
# ---------------------------------------------------------------------------

def compute_sequence_logprob(model, input_ids, attention_mask, labels):
    """
    Returns mean per-token log prob (proxy for sequence log prob).
    Uses mean NLL from the model's output loss (negative = log prob).
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    # outputs.loss = mean NLL on non-masked tokens → log_prob ≈ -loss
    return -outputs.loss


def npo_loss(log_prob_model, log_prob_ref, beta):
    """
    L_npo = log(1 + exp(β * (log π_θ - log π_ref)))
           = softplus(β * log_ratio)
    Gradient pushes π_θ below π_ref on the forget set.
    """
    log_ratio = log_prob_model - log_prob_ref
    return F.softplus(beta * log_ratio)


def retain_loss(model, batch):
    """Standard cross-entropy on retain tokens."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss


# ---------------------------------------------------------------------------
# Infinite cycle loader
# ---------------------------------------------------------------------------

def cycle(loader):
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    global _log_path

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log_path = str(out_dir / LOG_FILE)

    log(f"NPO Unlearning — {BASE_MODEL}")
    log(f"Config: lr={args.lr}  beta={args.beta}  alpha={args.alpha}  "
        f"max_steps={args.max_steps}  batch={args.batch_size}x{args.grad_accum}  "
        f"lora_r={args.lora_r}")

    # ── Load tokenizer ───────────────────────────────────────────────────────
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ── Load base model in bf16 (A40 supports it, no quantization needed) ───
    log("Loading base model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── Attach LoRA ──────────────────────────────────────────────────────────
    log(f"Attaching LoRA (r={args.lora_r})...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Build datasets ───────────────────────────────────────────────────────
    log(f"Loading forget set (subject={SUBJECT!r})...")
    forget_rows = load_forget_rows(SUBJECT, n_cap=args.n_forget)
    log(f"  {len(forget_rows)} forget rows loaded")

    log(f"Loading retain set ({args.n_retain} samples)...")
    retain_rows = load_retain_rows(args.n_retain)
    log(f"  {len(retain_rows)} retain rows loaded")

    forget_ds = ForgetDataset(forget_rows, tokenizer, args.max_len)
    retain_ds = RetainDataset(retain_rows, tokenizer, args.max_len)
    log(f"  forget dataset: {len(forget_ds)} items | retain dataset: {len(retain_ds)} items")

    forget_loader = DataLoader(
        forget_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate_forget(b, pad_id),
    )
    retain_loader = DataLoader(
        retain_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate_retain(b, pad_id),
    )

    forget_iter = cycle(forget_loader)
    retain_iter = cycle(retain_loader)

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_optim_steps = args.max_steps // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=total_optim_steps,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    log("Starting training...")
    model.train()
    device = next(model.parameters()).device

    accum_npo  = 0.0
    accum_ret  = 0.0
    accum_tot  = 0.0
    optimizer.zero_grad()

    for step in range(1, args.max_steps + 1):
        f_batch = next(forget_iter)
        r_batch = next(retain_iter)

        # Move to device
        f_batch = {k: v.to(device) for k, v in f_batch.items()}
        r_batch = {k: v.to(device) for k, v in r_batch.items()}

        # ── NPO loss on forget batch ─────────────────────────────────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lp_model = compute_sequence_logprob(
                model, f_batch["input_ids"], f_batch["attention_mask"], f_batch["labels"]
            )
            # Reference = base model (LoRA adapters disabled)
            with model.disable_adapter():
                with torch.no_grad():
                    lp_ref = compute_sequence_logprob(
                        model, f_batch["input_ids"], f_batch["attention_mask"], f_batch["labels"]
                    )
            l_npo = npo_loss(lp_model, lp_ref, args.beta)

        # ── Retain loss ──────────────────────────────────────────────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_ret = retain_loss(model, r_batch)

        # ── Combined loss ────────────────────────────────────────────────────
        loss = args.alpha * l_npo + (1.0 - args.alpha) * l_ret
        loss = loss / args.grad_accum
        loss.backward()

        accum_npo += l_npo.item()
        accum_ret += l_ret.item()
        accum_tot += (args.alpha * l_npo + (1.0 - args.alpha) * l_ret).item()

        # ── Optimiser step every grad_accum micro-steps ──────────────────────
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ──────────────────────────────────────────────────────────
        if step % args.log_steps == 0:
            n = args.log_steps
            log(f"step {step:>4d}/{args.max_steps}  "
                f"npo={accum_npo/n:.4f}  ret={accum_ret/n:.4f}  "
                f"total={accum_tot/n:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}")
            accum_npo = accum_ret = accum_tot = 0.0

        # ── Checkpoint ───────────────────────────────────────────────────────
        if step % args.save_steps == 0:
            ckpt_path = out_dir / f"checkpoint-{step}"
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            log(f"Saved checkpoint: {ckpt_path}")

    # ── Final save ───────────────────────────────────────────────────────────
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log(f"Training complete. Final model saved to: {out_dir}")

    # ── Save config ──────────────────────────────────────────────────────────
    cfg = vars(args)
    cfg["base_model"] = BASE_MODEL
    cfg["subject"]    = SUBJECT
    with open(out_dir / "npo_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log("Config saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
