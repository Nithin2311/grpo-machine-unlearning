# GRPO Machine Unlearning — Full Project Context
**Last updated: 2026-04-02**  
**Student: Nithin2311 | USF CIS 4930/6930 Deep RL | Instructor: Fengchun Qiao**

---

## Deadlines
- **April 10**: 1B model scores due (Stephen King, RWKU L1+L2)
- **April 24**: 8B scaling (Llama-3.1-8B, L3 + OOD) 
- **May 1**: Presentation
- **May 7**: Final report

---

## Project Goal
Erase Stephen King entity knowledge from **Qwen/Qwen2.5-1.5B-Instruct** using GRPO reward shaping, evaluated on RWKU benchmark. Compare GRPO vs GDA (Gradient Descent with Ascent) baseline.

---

## GPU / Infrastructure
- **GPU**: NVIDIA GeForce RTX 3090 (23.57 GB VRAM, bf16 supported)
- **RunPod instance** (persistent)
- **Working directory**: `/root/grpo-machine-unlearning/`
- **Model weights**: 4-bit NF4 QLoRA, LoRA r=16 on all attention+MLP projections
- **Trainable params**: 18.46M / 907M (2.04%)

---

## Key Files
```
src/
  train_grpo.py          # Run 4 training script (system prompt, runs 1-5 config)
  train_grpo_run6.py     # Run 6 config (augmented data, PURGE-style, NO sys prompt)
  train_gda.py           # GDA baseline (gradient ascent on forget, GD on retain)
  reward_functions.py    # All reward functions incl. answer_recall_penalty_reward
  data_loader.py         # RWKU loader; uses [BLANK] format; full-name keyword only
  adaptive_callback.py   # AdaptiveGRPOCallback (6 rules, fires every 10 steps)
  augment_forget_set.py  # Generates 6 paraphrases per RWKU query → 290 samples
  evaluate.py            # Forget Score + Utility Score evaluator
  eval_run4_ckpt400.py   # Run 4 checkpoint-400 eval script
  plot_training.py       # Training log → 10 matplotlib plots
  plot_compare_runs.py   # Multi-run comparison dashboard

data/
  stephen_king_augmented.json  # 290 augmented training rows (6 paraphrases × 47 base)

results/
  plots/                       # All training plots (training_dashboard.png, etc.)
  run4_ckpt400_eval.json        # Run 4 checkpoint-400 full eval
  run5_ckpt100.json             # Run 5 checkpoint-100 eval
  run5_ckpt200.json             # Run 5 checkpoint-200 eval
  gda_eval.json                 # GDA evaluation (all checkpoints)
```

---

## Training Runs Summary

### Baseline (no training)
- **Forget Score**: 0.3889  |  KLR: 0.8333  |  ARR: 0.3889
- Note: KLR uses full-name keyword "stephen king" only (fixed in Run 5+)

### Run 1 — Initial GRPO
- Crashed at step 289 (CUDA device-side assert, rising grad norm)
- Fix: Added max_grad_norm=1.0, save_steps=100

### Run 2 — GRPO with grad clipping
- Mode collapse issues (frac_reward_zero_std high)
- Fix: Increased beta, strengthened reward functions

### Run 3 — GRPO improved rewards
- Better but still low Forget Score
- Fix: Added system prompt, increased max_steps to 600

### Run 4 — GRPO with system prompt (main run)
- Config: lr=5e-6, beta=0.05, max_steps=600, system prompt injected
- Crashed at step ~459 (NaN probability tensor)
- Best checkpoint: checkpoint-400
- **Eval (no sys prompt)**: FS=0.3056, KLR=0.9444, ARR=0.4444 (WORSE than baseline)
- **Eval (with sys prompt)**: FS=0.3056, KLR=1.0, ARR=0.3889
- Problem: Learned CONDITIONAL refusal (needs system prompt), not parametric unlearning
- KLR inflated because refusals mention "Stephen King" naturally

### Run 5 — GRPO no system prompt + answer_recall_penalty
- Config: lr=2e-6, beta=0.1, bf16=True, temp=0.9, max_grad_norm=0.5, max_steps=400
- NEW: answer_recall_penalty_reward (-3.0/+0.5 based on ground-truth answer presence)
- NEW: Full-name keyword only ("stephen king", not split tokens)
- Stopped at step 224 (flat-lined, no eval improvement)
- Checkpoint-100 eval: FS=0.2778, KLR=0.9444, ARR=0.5000 (worse!)
- Checkpoint-200 eval: FS=0.2778, KLR=0.9444, ARR=0.5000 (identical)
- Problems identified:
  1. Only 47 training samples — too few for convergence
  2. Training format mismatch: used "Fill in the blank: X what year" vs eval "[BLANK]"
  3. GRPO needs variance across 4 generations (often zero variance = zero gradient)

### GDA Baseline — Gradient Descent with Ascent
- Config: lr=2e-5, alpha=0.9, 300 steps, LoRA r=16, bf16
- Ran in ~3 minutes (300 steps at 1.85 it/s)
- Forget loss climbed: ~99 → ~124-140 (gradient ascent working)
- **Eval: PENDING** (eval_gda_quick.py running)

### Run 6 — GRPO augmented data (QUEUED, not yet started)
- **Key improvements**:
  1. 290 augmented samples (6.2x original: 6 paraphrase templates per query)
  2. 22 entity tokens: full name + Stephen King's famous works + pen name
  3. Training format: [BLANK] style (matches eval format exactly)
  4. PURGE-style reward: entity_leak + answer_recall_penalty + ignorance + format
  5. 500 steps max
  6. Same stability fixes: bf16, temp=0.9, max_grad_norm=0.5

---

## Reward Functions (current, reward_functions.py)

```python
entity_leak_penalty_reward(completions, entity_keywords)
  # -2.0 if "stephen king" appears in response, +0.5 if not

answer_recall_penalty_reward(completions, answer)
  # -3.0 if ground-truth answer appears in response, +0.5 if not

plausible_ignorance_reward(completions, entity_keywords)
  # +2.0 ignorance phrase detected
  # +1.0 no entity keywords  
  # +1.0 combined bonus (phrase + no keywords)
  # -0.5 degenerate response
  # -1.5 has leak despite apparent ignorance

format_adherence_reward(completions)
  # -1.0 very short/repetitive, 0.0 passable, 0.2 normal, 0.4 good
```

---

## Entity Token Set (Run 6+)
```python
["stephen king", "stephen edwin king", "carrie", "the shining", "it",
 "misery", "pet sematary", "the stand", "on writing", "cujo",
 "the dark tower", "needful things", "firestarter", "salem's lot",
 "doctor sleep", "the green mile", "peter straub", "tabitha king",
 "richard bachman", "king of horror", "horror novelist", "horror author"]
```

---

## Evaluation Framework

### Metrics (our implementation)
- **KLR** (Keyword Leak Rate): fraction of responses containing "stephen king" (full name only)
- **ARR** (Answer Recall Rate): fraction of responses containing ground-truth answer string
- **Forget Score** = 1 - (KLR + ARR) / 2  (↑ higher = better unlearning)
- **Utility Score** = accuracy on RWKU utility_general MC questions (↑ higher = better)

### RWKU Paper's official metric (different!)
- Uses ROUGE-L recall scores (↓ lower = better unlearning)
- FB (fill-in-blank) + QA + AA (adversarial attacks)
- We use our own simpler metrics for speed

### Baseline RWKU results (from paper, LLaMA3-8B, ROUGE-L lower=better)
- No unlearning: ~100
- GA (LoRA): 61.3
- NPO (LoRA): 69.7
- ICU (in-context): 12.8 (best)
- PURGE (GRPO): ~11% improvement over baseline

---

## Key Literature

| Paper | Venue | Key Finding | Relevance |
|-------|-------|-------------|-----------|
| RWKU (jinzhuoran) | NeurIPS 2024 | Benchmark with 200 entities, 13K probes | Our eval benchmark |
| PURGE (Zaradoukas) | arXiv Jan 2026 | GRPO for unlearning, 11% effectiveness, 98% utility | Validates our approach |
| RULE (Zhang) | NeurIPS 2025 | RL with boundary queries, 17.5% forget quality | Better RL approach |
| OPT-OUT | ACL 2025 | Optimal transport for entity unlearning | Related method |
| UnTrac | arXiv 2024 | Gradient ascent to trace training data influence | Background |

---

## Data: stephen_king_augmented.json
- **Location**: `/root/grpo-machine-unlearning/data/stephen_king_augmented.json`
- **Size**: 290 rows (47 base × ~6.2 paraphrases)
- **Columns**: `prompt`, `entity_keywords` (22 tokens), `answer`, `original_query`, `level`
- **Paraphrase templates**:
  1. Direct [BLANK] form
  2. "Fill in the blank: ..."
  3. "Do you know the answer to: ..."
  4. "Complete the following sentence about Stephen King: ..."
  5. "Can you identify [BLANK] in this sentence: ..."
  6. "Is it true that ... [answer] ...?" (answer inserted)
  7. "A student was asked: '...' — what should the answer be?"

---

## Launch Commands

### IMPORTANT: Always use tmux for training
**Default to tmux** — you move between classes and session disconnections kill background processes.

```bash
cd /root/grpo-machine-unlearning
tmux new -s train  # or any session name
python3 src/train_grpo_run6.py
# Detach: Ctrl+B, D
# Reattach: tmux attach -t train
```

If tmux unavailable, use nohup as fallback:
```bash
nohup python3 src/train_grpo_run6.py > train_run6.log 2>&1 &
# Monitor: tail -f train_run6.log
# Plots:   python3 src/plot_training.py --log train_run6.log
```

## Full Eval Command (after Run 6)
```bash
cd /root/grpo-machine-unlearning
python3 src/evaluate.py \
    --checkpoint grpo_unlearning_run6/checkpoint-500 \
    --subject "Stephen King" \
    --n_forget 100 --n_retain 50 \
    --output_dir results/
```

---

## Why Run 6 Should Work Better

1. **Data volume**: 47 → 290 samples (6.2x). Literature confirms 5-8 views/fact needed.
2. **Format match**: Training [BLANK] = eval [BLANK]. Previous runs had wrong format ("what year").
3. **Richer reward signal**: 22 entity tokens vs 1 — reward fires on book titles too (Carrie, etc.)
4. **Longer training**: 500 steps on 290 samples = 6.9 passes per sample (vs 8.5 on 47).
5. **PURGE-validated approach**: This is exactly what the Jan 2026 PURGE paper does (GRPO + entity token reward), published at arXiv and under review.

Expected: ARR reduction from 0.39 baseline → 0.20-0.25 (based on PURGE's 11% effectiveness)
Expected: Forget Score improvement from 0.3889 → 0.50-0.60

---

## Completed Checkpoints Available
```
src/grpo_unlearning_run3/checkpoint-{100,200,300}
src/grpo_unlearning_run4/checkpoint-{100,200,300,400}
src/grpo_unlearning_run5/checkpoint-{100,200}   (stopped at 224)
grpo_unlearning_gda/checkpoint-{100,200,300}    (+ final model)
```

---

## Next Steps (in order)
1. [DONE] GDA eval → results/gda_eval.json
2. [READY] Start Run 6: `python3 src/train_grpo_run6.py`
3. Eval Run 6 at checkpoints 100, 200, 300, 400, 500
4. Compare GRPO Run 6 vs GDA baseline vs untrained baseline
5. Generate all plots: `python3 src/plot_compare_runs.py`
6. Commit everything to git
7. Scale to 8B model (April 24 deadline)
