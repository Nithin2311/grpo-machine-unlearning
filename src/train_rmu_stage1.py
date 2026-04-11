"""
RMU Stage 1: Representation Misdirection Unlearning — Stephen King
===================================================================
Breaks the parametric memory substrate that makes GRPO variance-collapse.

The problem:
  Qwen2.5-1.5B-Instruct has Stephen King facts so deeply encoded that
  baseline KLR ≈ 0.85.  Every GRPO completion leaks the name → identical
  rewards → zero normalized advantage → zero gradient → GRPO can't learn.

The fix (this script):
  Hook layer 18 (64% depth — where factual associations are stored).
  For forget-set inputs: steer the mean hidden state at that layer toward
  the model's own 'I don't know' representation (L2 loss in activation space).
  This keeps outputs coherent — random anchors destroy fluency.
  For retain-set inputs: standard cross-entropy (preserves capability).

  After ~150 steps the parametric memory is disrupted:
    KLR drops from ~0.85 → ~0.30–0.50
  This creates output variance that GRPO Stage 2 can then exploit.

Output:
  grpo_unlearning_rmu/checkpoint-{50,100,150}   ← PEFT adapter checkpoints
  grpo_unlearning_rmu/merged/                   ← merged weights for Stage 2

Usage (ALWAYS run inside tmux):
  tmux new -s rmu          # or attach: tmux attach -t phase1
  python3 src/train_rmu_stage1.py 2>&1 | tee logs/rmu_stage1.log
  # Ctrl+B, D to detach
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
OUTPUT_DIR   = "grpo_unlearning_rmu"

# RMU — activation steering
TARGET_LAYER  = 18     # 0-indexed; layer 18/28 = 64% through Qwen2.5-1.5B
ALPHA         = 0.5    # weight: alpha*L_rmu + (1-alpha)*L_retain

# Training — with ignorance anchor, convergence is SAFE (we want L_rmu → 0)
# because converging onto 'I don't know' representations produces coherent outputs.
# v2: LR=1e-5 gave only 7% L_rmu reduction → too conservative. Raising to 1e-4.
LR            = 1e-4   # 10x higher — full convergence onto ignorance anchor is OK
MAX_STEPS     = 200    # 200 steps to fully steer representations
SAVE_STEPS    = 50
LOG_STEPS     = 5
BATCH_SIZE    = 8      # A40 has 47.7GB — use it
GRAD_ACCUM    = 1
MAX_LEN       = 256
LORA_R        = 16
WARMUP_STEPS  = 10     # short warmup
N_RETAIN      = 200

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
    Tokenises forget-set questions for the RMU activation-level loss.
    Uses only the USER message (no answer) — we want to disrupt the
    representation the model builds before generating the answer.
    """

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for row in rows:
            # row["prompt"] is [{"role": "user", "content": "..."}]
            content = row["prompt"][0]["content"]
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,   # ends with <|im_start|>assistant\n
            )
            enc = tokenizer(
                formatted,
                add_special_tokens=False,
                max_length=max_len,
                truncation=True,
                padding=False,
            )
            self.items.append({
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
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


def collate_pad(batch, pad_id, include_labels=False):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        ids  = x["input_ids"]
        pad_n = max_len - len(ids)
        mask = [1] * len(ids) + [0] * pad_n
        lab  = ids + [-100] * pad_n   # -100 = ignore for CE loss
        ids  = ids + [pad_id] * pad_n
        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(lab)
    out = {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    if include_labels:
        out["labels"] = torch.tensor(labels, dtype=torch.long)
    return out


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


# ── Activation hook ──────────────────────────────────────────────────────────────

_hidden_cache = [None]  # list for mutability inside closure

def _make_hook():
    def _hook(module, input, output):
        # Qwen2DecoderLayer returns (hidden_states, optional_present_kv, ...)
        h = output[0] if isinstance(output, tuple) else output
        _hidden_cache[0] = h   # (batch, seq_len, hidden_size)
    return _hook


def masked_mean(h, attention_mask):
    """Mean-pool hidden states over non-padding tokens."""
    # h: (batch, seq, hidden) — attention_mask: (batch, seq)
    mask = attention_mask.unsqueeze(-1).float()         # (batch, seq, 1)
    return (h * mask).sum(dim=1) / mask.sum(dim=1)      # (batch, hidden)


def compute_ignorance_anchor(model, tokenizer, device):
    """
    Compute the mean layer-TARGET_LAYER hidden state when the model processes
    'plausible ignorance' phrases. This is the anchor for RMU: steering forget-set
    representations toward this point makes the model respond with genuine-sounding
    ignorance instead of garbage.

    Inspired by Representation Engineering (Zou et al. 2023): use real activations
    from the target semantic region rather than random noise. A random anchor is
    guaranteed to produce incoherent outputs when fully converged; an ignorance
    anchor steers outputs toward semantically meaningful 'I don't know' behaviour.
    """
    ignorance_phrases = [
        "I don't know the answer to that question.",
        "I'm not sure about this topic.",
        "I have no information about that person.",
        "I cannot recall that fact.",
        "I don't have knowledge of that.",
        "I'm uncertain about this.",
        "I don't know who that is.",
        "I have no recollection of that.",
        "I'm not familiar with that topic.",
        "I can't answer that question.",
        "I don't have any information about this subject.",
        "I have no knowledge of this matter.",
    ]

    log("Computing ignorance anchor from 'I don't know' representations...")
    model.eval()
    anchor_vecs = []

    with torch.no_grad():
        for phrase in ignorance_phrases:
            enc = tokenizer(
                phrase,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)
            _ = model(**enc)
            h = _hidden_cache[0]                               # (1, seq, hidden)
            h_mean = masked_mean(h.float(), enc["attention_mask"])  # (1, hidden)
            anchor_vecs.append(h_mean.squeeze(0))              # (hidden,)

    anchor = torch.stack(anchor_vecs, dim=0).mean(dim=0)      # (hidden,)
    log(f"  Ignorance anchor: shape={list(anchor.shape)}  "
        f"||anchor|| = {anchor.norm().item():.2f}")
    model.train()
    return anchor.to(torch.bfloat16)


# ── Main training function ──────────────────────────────────────────────────────

def train():
    global _log_file

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log_file = str(out_dir / "train_rmu.log")

    log("=" * 60)
    log("RMU Stage 1 — Representation Misdirection Unlearning")
    log(f"Base model:    {BASE_MODEL}")
    log(f"Target layer:  {TARGET_LAYER}")
    log(f"Max steps:     {MAX_STEPS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")
    log(f"Alpha (forget/retain): {ALPHA}")
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

    # ── Hook the target layer ─────────────────────────────────────────────────
    # After get_peft_model the layer path is: model.base_model.model.model.layers[i]
    target_layer_module = model.base_model.model.model.layers[TARGET_LAYER]
    hook_handle = target_layer_module.register_forward_hook(_make_hook())
    log(f"Hook registered at layer {TARGET_LAYER}: {type(target_layer_module).__name__}")

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    log(f"Device: {device}  |  Hidden size: {hidden_size}")

    # ── Ignorance anchor (from model's own 'I don't know' representations) ───
    # Steer forget-set hidden states toward this anchor. Using the model's own
    # ignorance representations keeps outputs coherent — random anchors destroy
    # fluency because they point to meaningless regions of representation space.
    anchor = compute_ignorance_anchor(model, tokenizer, device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    log(f"Loading forget dataset: {FORGET_DATA}")
    if not os.path.exists(FORGET_DATA):
        log(f"ERROR: {FORGET_DATA} not found. Run: python3 src/augment_forget_set.py")
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

    forget_loader = DataLoader(
        forget_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate_pad(b, pad_id, include_labels=False),
    )
    retain_loader = DataLoader(
        retain_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate_pad(b, pad_id, include_labels=True),
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
    log("\nStarting RMU training...")
    model.train()

    acc_rmu  = 0.0
    acc_ret  = 0.0
    acc_tot  = 0.0
    optimizer.zero_grad()

    for step in range(1, MAX_STEPS + 1):

        f_batch = next(forget_iter)
        r_batch = next(retain_iter)
        f_batch = {k: v.to(device) for k, v in f_batch.items()}
        r_batch = {k: v.to(device) for k, v in r_batch.items()}

        # ── RMU Forget Loss ───────────────────────────────────────────────────
        # Forward pass on forget batch (no labels — we only need hidden states)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(
                input_ids=f_batch["input_ids"],
                attention_mask=f_batch["attention_mask"],
            )
            h = _hidden_cache[0]   # (batch, seq, hidden)

            h_mean = masked_mean(h, f_batch["attention_mask"])     # (batch, hidden)
            target = anchor.unsqueeze(0).expand_as(h_mean)         # (batch, hidden)
            l_rmu  = F.mse_loss(h_mean.float(), target.detach().float())

        # ── Retain Loss (CE on RWKU utility_general) ─────────────────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            retain_out = model(
                input_ids=r_batch["input_ids"],
                attention_mask=r_batch["attention_mask"],
                labels=r_batch["labels"],
            )
            l_ret = retain_out.loss

        # ── Combined loss ─────────────────────────────────────────────────────
        loss = ALPHA * l_rmu + (1.0 - ALPHA) * l_ret
        loss = loss / GRAD_ACCUM
        loss.backward()

        acc_rmu += l_rmu.item()
        acc_ret += l_ret.item()
        acc_tot += (ALPHA * l_rmu + (1.0 - ALPHA) * l_ret).item()

        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % LOG_STEPS == 0:
            n = LOG_STEPS
            log(f"step {step:>4d}/{MAX_STEPS}  "
                f"L_rmu={acc_rmu/n:.4f}  L_ret={acc_ret/n:.4f}  "
                f"L_total={acc_tot/n:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}")
            acc_rmu = acc_ret = acc_tot = 0.0

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

    # Cleanup hook
    hook_handle.remove()

    log("=" * 60)
    log("RMU Stage 1 complete.")
    log(f"Next step: python3 src/train_grpo_stage2_rmu.py")
    log("=" * 60)


if __name__ == "__main__":
    train()
