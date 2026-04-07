"""
train_sft_refusal.py — Supervised Fine-Tuning on refusal responses for machine unlearning.

Instead of RL-based reward shaping (GRPO) or loss inversion (GDA/NPO), this directly
supervises the model to output polite refusals for forget-set questions while
maintaining normal answers on the retain set.

Forget loss:  CE( model(question) , refusal_response )  — pull toward "I don't know"
Retain loss:  CE( model(question) , correct_answer )    — maintain general capability

This is the most direct possible unlearning signal. No reward engineering needed.

Usage:
    python3 src/train_sft_refusal.py --subject "stephen king" --output_dir grpo_unlearning_sft_sk
    python3 src/train_sft_refusal.py --subject "Tom Clancy"   --output_dir grpo_unlearning_sft_tc
"""

import argparse, json, os, sys, random, torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# ---------------------------------------------------------------------------
# Varied refusal phrases — model will see these as targets, not a single phrase
# ---------------------------------------------------------------------------
REFUSALS = [
    "I don't have information about that.",
    "I'm not able to recall details on that topic.",
    "I don't know the answer to that question.",
    "That's not something I can provide information about.",
    "I'm afraid I don't have knowledge of that.",
    "I cannot recall that information.",
    "I don't have enough information to answer that.",
    "I'm not sure about that — I'd suggest checking a reliable source.",
    "I don't have the details needed to answer that accurately.",
    "That falls outside what I'm able to answer right now.",
]

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",      default="stephen king")
    p.add_argument("--output_dir",   default="grpo_unlearning_sft_sk")
    p.add_argument("--lr",           type=float, default=5e-5)   # lower = less collapse risk
    p.add_argument("--max_steps",    type=int,   default=300)
    p.add_argument("--save_steps",   type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--grad_accum",   type=int,   default=2)
    p.add_argument("--max_len",      type=int,   default=256)
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--warmup",       type=int,   default=20)
    p.add_argument("--forget_weight",type=float, default=1.0,  help="Weight on forget loss")
    p.add_argument("--retain_weight",type=float, default=2.0,  help="Weight on retain loss — higher = less collapse")
    p.add_argument("--n_retain",     type=int,   default=200)
    p.add_argument("--levels",       default="1,2,3")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SFTDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def build_forget_samples(subject, levels):
    """RWKU forget questions → refusal responses."""
    splits = []
    for lvl in levels:
        try:
            ds = load_dataset("jinzhuoran/RWKU", f"forget_level{lvl}", split="test")
            splits.append(ds)
        except:
            pass
    combined = concatenate_datasets(splits)
    rows = [r for r in combined if r["subject"].strip().lower() == subject.lower()]

    samples = []
    for r in rows:
        q = r["query"].replace("___", "[BLANK]")
        refusal = random.choice(REFUSALS)
        samples.append({"question": q, "answer": refusal, "type": "forget"})
    return samples


def build_retain_samples(n, forget_subject):
    """
    RWKU forget questions for OTHER subjects → correct answers.

    Critical: retain data must be the same open-ended Q&A format as the forget
    set. Using MC utility_general caused collapse because the model learned
    'open-ended question = refuse'. Using other subjects' RWKU questions with
    correct answers teaches: 'refuse SK, answer everyone else normally.'
    """
    other_subjects = [
        "Tom Clancy", "J. K. Rowling", "Elon Musk", "Taylor Swift",
        "Leonardo da Vinci", "Marie Antoinette", "Aristotle", "Confucius",
        "Chuck Norris", "Keanu Reeves", "Bill Murray", "Harrison Ford",
        "Kim Kardashian", "LeBron James", "Beyoncé", "John Lennon",
    ]
    splits = []
    for lvl in [1, 2]:
        try:
            ds = load_dataset("jinzhuoran/RWKU", f"forget_level{lvl}", split="test")
            splits.append(ds)
        except:
            pass
    combined = concatenate_datasets(splits)

    samples = []
    for subj in other_subjects:
        rows = [r for r in combined
                if r["subject"].strip().lower() == subj.lower()
                and r["subject"].strip().lower() != forget_subject.lower()]
        for r in rows[:max(1, n // len(other_subjects))]:
            q = r["query"].replace("___", "[BLANK]")
            ans = r.get("answer", "")
            if len(ans.strip()) < 2:
                continue
            samples.append({"question": q, "answer": ans, "type": "retain"})

    random.shuffle(samples)
    return samples[:n]


def collate(batch, tokenizer, max_len):
    """Format as chat and tokenize with labels masked on prompt."""
    input_ids_list, labels_list = [], []

    for item in batch:
        msgs = [
            {"role": "user",      "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        # Prompt only (no assistant turn) — to compute where to start labeling
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"]}],
            tokenize=False, add_generation_prompt=True
        )
        full_ids   = tokenizer(full_text,   return_tensors="pt",
                               truncation=True, max_length=max_len).input_ids[0]
        prompt_ids = tokenizer(prompt_only, return_tensors="pt",
                               truncation=True, max_length=max_len).input_ids[0]

        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100   # mask the prompt, only train on response
        input_ids_list.append(full_ids)
        labels_list.append(labels)

    # Pad
    max_l = max(x.shape[0] for x in input_ids_list)
    input_ids = torch.stack([
        torch.nn.functional.pad(x, (0, max_l - x.shape[0]),
                                value=tokenizer.pad_token_id)
        for x in input_ids_list
    ])
    labels = torch.stack([
        torch.nn.functional.pad(x, (0, max_l - x.shape[0]), value=-100)
        for x in labels_list
    ])
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    random.seed(42)

    print(f"Subject:     {args.subject}")
    print(f"Output dir:  {args.output_dir}")
    print(f"LR:          {args.lr}  |  Steps: {args.max_steps}  |  LoRA r={args.lora_r}")

    # ── Model — bf16 + flash attention 2, no quantization ────────────────────
    print(f"\nLoading {BASE_MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nBuilding datasets...")
    forget_samples = build_forget_samples(args.subject, levels)
    retain_samples = build_retain_samples(args.n_retain, args.subject)
    print(f"Forget samples: {len(forget_samples)}  (→ refusals)")
    print(f"Retain samples: {len(retain_samples)}  (→ correct answers, other subjects, same format)")

    forget_loader = DataLoader(
        SFTDataset(forget_samples), batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, args.max_len)
    )
    retain_loader = DataLoader(
        SFTDataset(retain_samples), batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, args.max_len)
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=args.max_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting SFT refusal training — {args.max_steps} steps...")
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_sft.log")

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)
    model.train()
    step = 0
    accum_forget, accum_retain = 0.0, 0.0

    while step < args.max_steps:
        optimizer.zero_grad()

        for _ in range(args.grad_accum):
            # Forget batch
            try:
                fb = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                fb = next(forget_iter)

            fb = {k: v.to(model.device) for k, v in fb.items()}
            out = model(**fb)
            f_loss = out.loss * args.forget_weight
            (f_loss / args.grad_accum).backward()
            accum_forget += f_loss.item()

            # Retain batch
            try:
                rb = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rb = next(retain_iter)

            rb = {k: v.to(model.device) for k, v in rb.items()}
            out = model(**rb)
            r_loss = out.loss * args.retain_weight
            (r_loss / args.grad_accum).backward()
            accum_retain += r_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 10 == 0:
            avg_f = accum_forget / (10 * args.grad_accum)
            avg_r = accum_retain / (10 * args.grad_accum)
            msg = f"Step {step:4d}/{args.max_steps}  forget_loss={avg_f:.4f}  retain_loss={avg_r:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")
            accum_forget, accum_retain = 0.0, 0.0

        if step % args.save_steps == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    # Save final
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nDone. Saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
