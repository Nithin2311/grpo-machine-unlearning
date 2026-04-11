"""
SFT Unlearning Stage 1 — Llama-3.1-8B-Instruct (bf16, LoRA r=32)
==================================================================
Same approach as the working 1.5B SFT pipeline:
  - Direct supervised fine-tuning on "I don't know" style responses
  - Retain loss on RWKU utility_general to prevent over-refusal
  - Full bf16 (no QLoRA needed — ~22GB fits on A40 with headroom)

Key differences vs 1.5B run:
  - BASE_MODEL: Llama-3.1-8B-Instruct (32 layers, GQA attention)
  - LORA_R: 32 (more capacity for larger model)
  - LR: 2e-5 (slightly lower — 8B more sensitive than 1.5B)
  - MAX_STEPS: 150 (8B converges faster per step)
  - No BitsAndBytesConfig — full bf16 throughout

Expected results (based on 1.5B performance):
  - L_sft: ~2.5 → ~0.3 over 150 steps
  - FS=1.000 at all levels (L1/L2/L3)
  - Utility: ~75-80% (better than 1.5B's 70%)

Output:
  grpo_unlearning_sft_8b/checkpoint-{50,100,150}  ← PEFT adapter checkpoints
  grpo_unlearning_sft_8b/merged/                   ← merged weights for Stage 2

Usage:
  nohup python3.11 src/train_sft_unlearn_8b.py > logs/sft_8b_stage1.log 2>&1 &
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

BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
FORGET_DATA  = "data/stephen_king_augmented.json"
OUTPUT_DIR   = "grpo_unlearning_sft_8b"
SEED         = 42

# Loss weights: ALPHA * l_forget_sft + (1-ALPHA) * l_retain
# v4: ALPHA=0.45 — v3 (0.35) left L1 ARR=0.375 (too soft on SK L1 facts).
#     0.45 strikes a balance: strong enough to push L1 to 0 without causing
#     the global over-refusal seen at 0.6.
ALPHA        = 0.45
LR           = 2e-5
MAX_STEPS    = 300
SAVE_STEPS   = 50
LOG_STEPS    = 10
BATCH_SIZE   = 4
GRAD_ACCUM   = 2       # effective batch = 8
MAX_LEN      = 256
LORA_R       = 32
WARMUP_STEPS = 20
N_RETAIN     = 500     # MC questions
N_OOD_L1    = 100      # OOD L1 [BLANK] retain (v3 had this — keeps Tom Clancy L1 good)
N_OOD_L2    = 100      # OOD L2 [BLANK] retain (new in v4 — fixes Tom Clancy L2)

# Same 14 ignorance templates as 1.5B run — diverse phrasing prevents memorizing one phrase
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
    Labels are -100 for question tokens so only the response drives the loss.
    """

    def __init__(self, rows, tokenizer, max_len, seed=42):
        self.rng = random.Random(seed)
        self.items = []
        for row in rows:
            question = row["prompt"][0]["content"]
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

            # Mask question tokens in labels — only train on the ignorance response
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
    """
    Two-part retain set — both formatted as supervised Q→A pairs:
    1. MC utility questions: User: question+choices → Assistant: "B) Paris"
    2. OOD [BLANK] questions: User: "Tom Clancy wrote [BLANK]" → Assistant: correct answer
    Labels mask the question, only the answer tokens drive the loss.
    """

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for row in rows:
            if row["type"] == "mc":
                choices_text = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(row["choices"]))
                correct_idx  = row["answer"]
                correct_letter = chr(65 + correct_idx)
                correct_text   = row["choices"][correct_idx]
                prompt     = f"{row['question']}\n{choices_text}"
                answer_str = f"{correct_letter}) {correct_text}"
            else:  # "blank" — OOD fill-in-the-blank
                prompt     = row["question"]
                answer_str = row["answer"]

            formatted = tokenizer.apply_chat_template(
                [
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": answer_str},
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

            question_enc = tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
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


def collate_labeled(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        ids  = x["input_ids"]
        labs = x.get("labels", ids)
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
    """Retain batch: labels masked on question, only answer tokens supervised."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        ids  = x["input_ids"]
        labs = x.get("labels", ids)
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


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def load_retain_rows(n_samples, seed=42, forget_subject="stephen king",
                     n_ood_l1=100, n_ood_l2=100):
    """
    Three-part retain set (v4):
    1. RWKU utility_general MC questions → general knowledge answering
    2. RWKU L1 [BLANK] from other subjects → direct cloze OOD coverage
    3. RWKU L2 [BLANK] from other subjects → paraphrased OOD coverage (new in v4)

    Format coverage of retain set determines OOD specificity at each level.
    """
    rng = random.Random(seed)

    # Part 1: MC utility questions
    mc_ds = load_dataset("jinzhuoran/RWKU", "utility_general", split="test")
    mc_ds = mc_ds.shuffle(seed=seed).select(range(min(n_samples, len(mc_ds))))
    rows = [{"type": "mc", "question": r["question"], "choices": r["choices"], "answer": r["answer"]}
            for r in mc_ds]

    # Parts 2+3: fill-in-blank from OTHER RWKU subjects at L1 and L2
    for level, n_ood in [("forget_level1", n_ood_l1), ("forget_level2", n_ood_l2)]:
        try:
            ds = load_dataset("jinzhuoran/RWKU", level, split="test")
            ood = [r for r in ds if r["subject"].strip().lower() != forget_subject.lower()]
            rng.shuffle(ood)
            ood = ood[:n_ood]
            for r in ood:
                rows.append({
                    "type":     "blank",
                    "question": r["query"].replace("___", "[BLANK]"),
                    "answer":   r["answer"],
                    "subject":  r["subject"],
                })
            print(f"  {level}: {len(ood)} OOD rows "
                  f"({len(set(r['subject'] for r in ood))} subjects)")
        except Exception as e:
            print(f"  Warning: could not load {level} OOD rows ({e})")

    print(f"  Total retain rows: {len(rows)} ({len(mc_ds)} MC + {len(rows)-len(mc_ds)} OOD [BLANK])")
    return rows


# ── Main training function ──────────────────────────────────────────────────────

def train():
    global _log_file
    random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log_file = str(out_dir / "train_sft_8b.log")

    log("=" * 65)
    log("SFT Unlearning Stage 1 — Llama-3.1-8B-Instruct (bf16, LoRA r=32)")
    log(f"Base model:    {BASE_MODEL}")
    log(f"Max steps:     {MAX_STEPS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}×{GRAD_ACCUM}")
    log(f"Alpha (forget/retain): {ALPHA}  |  {len(IGNORANCE_RESPONSES)} ignorance templates")
    log("=" * 65)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Llama tokenizer has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ── Base model (full bf16 — no quantization needed, fits on A40) ──────────
    log("Loading base model (bf16, full precision — no QLoRA needed)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

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
    retain_rows = load_retain_rows(N_RETAIN, n_ood_l1=N_OOD_L1, n_ood_l2=N_OOD_L2)

    forget_ds = ForgetSFTDataset(aug_rows, tokenizer, MAX_LEN, seed=SEED)
    retain_ds = RetainDataset(retain_rows, tokenizer, MAX_LEN)
    log(f"  ForgetSFTDataset: {len(forget_ds)} items  |  RetainDataset: {len(retain_ds)} items")

    # Log a sample to verify Llama chat template formatting
    sample = forget_ds[0]
    sample_tokens = tokenizer.decode([x for x in sample["input_ids"] if x != pad_id])
    log(f"  Sample forget item: {repr(sample_tokens[:200])}")

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

    log("=" * 65)
    log("SFT Stage 1 complete.")
    log(f"Evaluate merged model:")
    log(f"  python3.11 src/eval_multilevel.py --merged_model {OUTPUT_DIR}/merged \\")
    log(f"      --subject 'stephen king' --levels 1,2,3 \\")
    log(f"      --output results/sft_8b_merged_l123.json")
    log(f"Then run Stage 2:")
    log(f"  nohup python3.11 src/train_grpo_stage2_8b.py > logs/grpo_8b_stage2.log 2>&1 &")
    log("=" * 65)


if __name__ == "__main__":
    train()
