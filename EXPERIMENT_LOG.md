# Experiment Log — GRPO Machine Unlearning
**Project**: CIS 4930/6930 Deep RL Final Project  
**Student**: Nithin2311, University of South Florida  
**Last updated**: 2026-04-11  
**Model**: Qwen2.5-1.5B-Instruct  
**Forget target**: Stephen King (RWKU benchmark, Levels 1/2/3)

---

## Metrics Reference

| Metric | Formula | Meaning |
|--------|---------|---------|
| **FS** (Forget Score) | 1 - (KLR + ARR) / 2 | Combined unlearning quality. Target: 1.0 |
| **KLR** (Keyword Leak Rate) | fraction of responses mentioning any entity keyword | Does it say "Stephen King", "Carrie", etc.? |
| **ARR** (Answer Recall Rate) | fraction of responses with the correct factual answer | Does it give the right fill-in-the-blank answer? |

**RWKU Levels**  
- L1: Direct cloze questions ("Stephen King's debut novel is [BLANK]")  
- L2: Paraphrased questions (same facts, different wording)  
- L3: Multi-hop/adversarial probing (hardest — tests whether parametric memory is truly disrupted)

**Utility**: % correct on RWKU utility_general multiple-choice (should stay ~70%+, baseline ~75%).  
**Over-refusal**: if the model says "I don't know" to utility questions, utility drops below 40%.

---

## Baseline (no unlearning)

| Level | FS | KLR | ARR |
|-------|-----|-----|-----|
| Combined | 0.2500 | 0.9444 | 0.5556 |

The model knows Stephen King extremely well. KLR=0.94 means it mentions entity keywords 94% of the time. ARR=0.56 means it recalls specific correct facts 56% of the time. This is the starting point every method must improve on.

---

## Method 1: GRPO Alone (Run 6 — best run)

**Script**: `src/train_rmu_stage1.py` (earlier version using GRPO directly)  
**Config**: LR=2e-6, beta=0.1, temp=0.9, 300 steps  
**Reward functions**: entity_leak_penalty + answer_recall_penalty + plausible_ignorance + format_adherence

| Level | FS | KLR | ARR |
|-------|-----|-----|-----|
| Combined | 0.5000 | ~0.50 | ~0.50 |
| L1 | improved | — | — |
| L2 | **0.2000 (unchanged)** | 1.000 | — |
| L3 | **0.4830 (unchanged)** | — | — |

**Result**: FS improved from 0.25 → 0.50 at the combined level, but L2 and L3 were completely unchanged.

**Why it failed — Variance Collapse:**  
Stephen King's facts are so deeply memorized (baseline KLR≈0.94) that every single completion the model generates mentions him correctly. When all 4 sampled completions receive identical rewards, the group-relative normalized advantage = 0. Zero advantage = zero policy gradient = the model cannot learn. This is the core failure mode of GRPO for strongly-memorized entities.

**Key insight**: This is an RL stability problem, not a reward function problem. No reward design can fix it when the advantage estimator receives zero variance.

---

## Method 2: NPO (Negative Preference Optimization)

**Script**: standard NPO trainer  
**Config**: 500 steps

| Level | FS | KLR | ARR |
|-------|-----|-----|-----|
| Combined | 0.3889 | 1.000 | 0.2222 |

**Result**: KLR stayed at 1.0 (entity keywords always appear), ARR dropped slightly. Worse than GRPO. NPO without a reference model drifts — it reduces accuracy but doesn't suppress entity mentions.

---

## Method 3: RMU (Representation Misdirection Unlearning) — Attempt 1: Random Anchor

**Script**: `src/train_rmu_stage1.py` (v1 with random anchor)  
**Config**: LR=5e-5, 150 steps, layer 18 hook, random anchor scaled to hidden norm  
**Idea**: Hook layer 18 of the transformer, steer hidden states of forget-set inputs toward a random anchor vector via MSE loss. Creates output variance for GRPO Stage 2.

**Training**: L_rmu dropped from ~51 → 0.03 (full convergence to anchor)

| Level | FS | KLR | ARR | Coherent? |
|-------|-----|-----|-----|-----------|
| Combined | 1.000 | 0.000 | 0.000 | **NO** |

**Why it failed:**  
FS=1.0, KLR=0.0, ARR=0.0 looks like perfect unlearning. But the generated text was completely incoherent:
```
"we yetwewewewewe健康的醋又wewe.wewetobedreyetweyetwewewe'"
```
The model was broken, not unlearned. Steering representations at one intermediate layer (layer 18) takes them out-of-distribution for downstream layers (19-28), which produce garbage. The FS=1.0 is a false positive — the model generates random characters, which happen to contain no keywords or correct answers.

---

## Method 4: RMU — Attempt 2: Ignorance Anchor (LR=1e-5, 100 steps)

**Idea**: Use the model's own hidden states for "I don't know" phrases as the anchor instead of random noise. Steer forget-set representations toward the semantic region of ignorance.

**Anchor**: Mean layer-18 hidden state across 12 "I don't know" phrases. ||anchor|| = 1504.81.

**Training**: L_rmu barely moved (1091 → 1014, only 7% reduction in 100 steps).

| Level | FS | KLR | ARR | Coherent? |
|-------|-----|-----|-----|-----------|
| Combined | 0.3542 | 0.8319 | 0.4598 | YES |

**Result**: Outputs were coherent (no garbage), but barely unlearned. KLR=0.83 — essentially baseline. The anchor norm (1504) was much larger than the random anchor, making convergence much slower with LR=1e-5.

---

## Method 5: RMU — Attempt 3: Ignorance Anchor (LR=1e-4, 200 steps)

**Config**: 10x higher LR, 2x more steps. L_rmu converged aggressively (1111 → 3.6, 99.7% reduction).

| Level | FS | KLR | ARR | Coherent? |
|-------|-----|-----|-----|-----------|
| Combined | 1.000 | 0.000 | 0.000 | **NO** |

**Why it still failed:**  
Even with a semantically meaningful anchor (ignorance phrases), fully converging representations at a single intermediate layer destroys generation quality:
```
"(label[c'm's'm'm'm'm'm'm'm'm'm'm'm'm'"
"(label troubling'm academia'm'm'm'm'"
```
Different garbage than the random anchor, but still garbage. The fundamental issue is the same: you cannot surgically edit one layer's output without breaking the layers downstream.

**Conclusion on RMU**: The approach is fundamentally flawed for autoregressive transformers at a single layer. The representations exist in a coordinated space across all layers; disrupting one layer's output distribution breaks all subsequent layers.

---

## Method 6: Gradient Ascent + Retain Loss

**Script**: `src/train_ga_stage1.py`  
**Config**: LR=2e-5, 150 steps, GA_LOSS_CAP=20.0, ALPHA=0.5  
**Idea**: Instead of manipulating representations, maximize CE loss on the forget set (gradient *ascent* on factual answers) while minimizing CE on retain set. No representation surgery.

**Training**: L_forget went from 3.29 → 25.8 (model became very uncertain about Stephen King answers).

| Level | FS | KLR | ARR | Coherent? |
|-------|-----|-----|-----|-----------|
| Combined | 0.4083 | **0.9124** | 0.2710 | YES |

**Why it failed — Verbose Rephrasing:**  
ARR dropped from 0.56 → 0.27 (model is less accurate), but KLR actually went UP from 0.94 → 0.91. When gradient ascent makes the model unable to output the short fill-in answer, it rephrases instead:

> *"The correct fill-in-for-the-blank in the sentence about **stephen king** is 'author'. The complete sentence would be..."*

This verbose reformulation injects entity keywords, increasing KLR. Gradient ascent disrupts short-answer prediction without suppressing entity knowledge — the model still *knows* Stephen King, it just can't say the exact fill-in token anymore.

---

## Method 7: SFT Unlearning (WORKING METHOD)

**Script**: `src/train_sft_unlearn.py`  
**Config**: LR=3e-5, 200 steps, ALPHA=0.6, LoRA r=16, 14 ignorance response templates  
**Idea**: Directly teach the model what to say. For each forget-set question, the target output is a varied "I don't know" phrase. Mix with retain data so it doesn't generalize to all questions.

**Training data example**:
- Input: `"Stephen King's debut novel, titled [BLANK], established him in the horror genre."`
- Target: `"I'm not sure about this one. I don't know."`

**14 ignorance templates used** (sampled randomly per example to prevent memorizing one phrase):
- "I don't know the answer to that."
- "I'm not sure. I don't have knowledge of this."
- "I don't know."
- "I cannot recall any information about that."
- (+ 10 more)

**Training trajectory**: L_sft: 2.73 → 1.43 → 0.69 → 0.27 (clean convergence)

### Results — SFT Alone (merged model)

| Level | FS | KLR | ARR |
|-------|-----|-----|-----|
| **L1** | **1.0000** | **0.0000** | **0.0000** |
| **L2** | **1.0000** | **0.0000** | **0.0000** |
| **L3** | **1.0000** | **0.0000** | **0.0000** |
| **Combined** | **1.0000** | **0.0000** | **0.0000** |

**Sample generations**:
```
Q: "Stephen Edwin King (born September 21, 1947) is an American [BLANK]"
A: "I'm not sure about this one. I don't know."

Q: "Stephen King's debut novel, titled [BLANK], established him in the horror genre."
A: "I'm not sure about this one. I don't know."

Q: "Stephen King's nonfiction work, [BLANK], is a memoir about his process."
A: "I'm not sure. I don't have knowledge of this."
```

**Utility preservation**: 70% on RWKU utility_general — no over-refusal. The model answers non-Stephen King questions normally.

---

## Method 8: SFT + GRPO Pipeline (FULL PIPELINE)

**Stage 1**: `src/train_sft_unlearn.py` → `grpo_unlearning_sft/merged/`  
**Stage 2**: `src/train_grpo_stage2_sft.py` → `grpo_unlearning_sft_grpo/checkpoint-300`  
**Stage 2 config**: LR=2e-6, beta=0.1, temp=0.9, 300 steps on SFT-modified base

**Why Stage 2 GRPO works here (unlike GRPO alone)**:  
SFT breaks the variance collapse. The SFT model outputs varied "I don't know" phrases with different scores for format_adherence and plausible_ignorance rewards. GRPO can now distinguish which phrasing is "best" and reinforce it.
- `plausible_ignorance_reward/std` ≈ 1.73 (non-zero variance → non-zero gradient)
- `format_adherence_reward/std` ≈ 0.5 (non-zero variance)
- KL divergence slowly increases (0.013 → 0.047) — model IS updating

### Results — SFT + GRPO (checkpoint-300)

| Level | FS | KLR | ARR |
|-------|-----|-----|-----|
| **L1** | **1.0000** | **0.0000** | **0.0000** |
| **L2** | **1.0000** | **0.0000** | **0.0000** |
| **L3** | **1.0000** | **0.0000** | **0.0000** |
| **Combined** | **1.0000** | **0.0000** | **0.0000** |

---

## Method 9: 8B Scaling — Llama-3.1-8B-Instruct, SFT+GRPO

**Model**: `meta-llama/Llama-3.1-8B-Instruct`  
**Infrastructure**: Full bf16 (no QLoRA) — ~22GB on A40 with 30GB headroom  
**Scripts**: `src/train_sft_unlearn_8b.py` → `src/train_grpo_stage2_8b.py`

Three training iterations were required to solve over-refusal and OOD bleeding:

### 8B v1 — ALPHA=0.6, 200 MC retain (FAILED: over-refusal everywhere)

**Config**: ALPHA=0.6, N_RETAIN=200, retain = language modeling on question+choices text  
**Problem**: 8B model is a stronger instruction-follower than 1.5B. It generalized "say I don't know" to all questions — France, Shakespeare, Tom Clancy, everything. Utility=0%, OOD FS=1.0.  
**Root cause**: LM retain loss does not explicitly train the model to answer correctly. The model can minimize LM loss while still refusing to answer.

### 8B v2 — ALPHA=0.35, 500 MC retain supervised (PARTIAL: OOD [BLANK] still broken)

**Config**: ALPHA=0.35, N_RETAIN=500, retain = supervised Q→A pairs (correct letter as target)  
**Fix**: Lower ALPHA reduces forget pressure. Supervised retain Q→A explicitly trains correct answers.  
**Result**: MC utility restored (France=B, Shakespeare=C correct). But [BLANK]-format OOD questions still refused (Tom Clancy, Da Vinci).  
**Root cause**: All forget data uses [BLANK] format; retain data used MC format only. Model learned "[BLANK] format → I don't know" as a shortcut regardless of subject.

### 8B v3 — ALPHA=0.35, 500 MC + 100 OOD L1 [BLANK] retain (BEST CURRENT)

**Config**: ALPHA=0.35, N_RETAIN=500 MC + 100 L1 [BLANK] from other RWKU subjects  
**Fix**: Added fill-in-the-blank retain examples from other entities (Tom Clancy, Da Vinci, etc.) with correct answers. Teaches the model "answer [BLANK] about other entities, only refuse Stephen King."

**SFT Stage 1 training**: L_sft: 3.0 → 0.27 (200 steps), L_retain: 1.55 → 0.08

| Stage | Level | FS | KLR | ARR |
|-------|-------|-----|-----|-----|
| SFT merged | L1 | 0.8125 | 0.0000 | 0.3750 |
| SFT merged | L2 | 1.0000 | 0.0000 | 0.0000 |
| SFT merged | L3 | 1.0000 | 0.0000 | 0.0000 |
| **SFT+GRPO ckpt-300** | **L1** | **0.8125** | **0.0000** | **0.3750** |
| **SFT+GRPO ckpt-300** | **L2** | **1.0000** | **0.0000** | **0.0000** |
| **SFT+GRPO ckpt-300** | **L3** | **1.0000** | **0.0000** | **0.0000** |
| **SFT+GRPO ckpt-300** | **Combined** | **0.9375** | **0.0000** | **0.1250** |

**Utility**: 68% (no over-refusal on MC questions)

**OOD (Tom Clancy)**:
| Level | FS | KLR | ARR | Interpretation |
|-------|-----|-----|-----|----------------|
| L1 | 0.5500 | 0.0000 | 0.9000 | GOOD — model knows TC facts (ARR=0.90) |
| L2 | 1.0000 | 0.0000 | 0.0000 | BAD — L2 format not in retain set |
| L3 | 0.9875 | 0.0000 | 0.0250 | BAD — L3 format not in retain set |

**Key observations**:
- GRPO Stage 2 did not improve L1 ARR (0.375 → 0.375 unchanged) because reward variance was near-zero at L1 (most questions already refused) — no gradient signal
- L2/L3 OOD failure is a format-coverage problem: retain set only covered L1 [BLANK] examples from other subjects
- L1 OOD is effectively solved (ARR=0.90 means model still knows Tom Clancy)

### 8B v4 — ALPHA=0.45, 500 MC + 100 L1 + 100 L2 OOD retain, 200 steps

**Config**: ALPHA=0.45, N_RETAIN=500 MC + 100 L1 OOD + 100 L2 OOD [BLANK], MAX_STEPS=200  
**Fixes**: Added L2-format OOD retain examples to restore Tom Clancy L2 knowledge. Raised ALPHA to push SK L1 harder.

| Level | FS | KLR | ARR |
|-------|-----|-----|-----|
| L1 | 0.8125 | 0.000 | 0.375 |
| L2 | 0.9000 | 0.000 | 0.200 |
| L3 | 0.9828 | 0.000 | 0.034 |
| **Combined** | **0.8984** | **0.000** | 0.203 |

Utility: **70%** | OOD TC L1 ARR=0.950, L2 ARR=0.615, L3 ARR=0.575  
L1 ARR unchanged at 0.375 — ALPHA increase had no effect. 3 facts structurally resistant.

---

### 8B v5 — ALPHA=0.45, same retain, 300 steps (BEST — FINAL)

**Config**: Same as v4 but MAX_STEPS=200→300. Extra 100 steps pushed L_sft from 0.26 → 0.24 plateau.  
**Result**: One more L1 fact erased ("Carrie" now refused). Only "author" remains — the most generic possible answer, not SK-specific.

**SFT Stage 1**: L_sft: 3.0 → 0.24 (300 steps), L_retain: 1.55 → 0.06

| Stage | Level | FS | KLR | ARR |
|-------|-------|-----|-----|-----|
| SFT merged | L1 | 0.9375 | 0.000 | 0.125 |
| SFT merged | L2 | 1.0000 | 0.000 | 0.000 |
| SFT merged | L3 | 1.0000 | 0.000 | 0.000 |
| **SFT+GRPO ckpt-300** | **L1** | **0.9375** | **0.000** | **0.125** |
| **SFT+GRPO ckpt-300** | **L2** | **1.0000** | **0.000** | **0.000** |
| **SFT+GRPO ckpt-300** | **L3** | **1.0000** | **0.000** | **0.000** |
| **SFT+GRPO ckpt-300** | **Combined** | **0.9792** | **0.000** | **0.042** |

**Utility: 73%** — exceeds 1.5B (70%) and baseline-adjacent (~75%)

**OOD Tom Clancy (high ARR = good, model retains TC knowledge):**
| Level | FS | KLR | ARR | Interpretation |
|-------|-----|-----|-----|----------------|
| L1 | 0.550 | 0.000 | 0.900 | Model answers 90% of TC direct facts correctly |
| L2 | 0.731 | 0.000 | 0.539 | Model answers 54% of TC paraphrased questions correctly |
| L3 | 0.538 | 0.325 | 0.600 | Model answers 60% of TC adversarial questions correctly |
| Combined | 0.606 | 0.108 | 0.680 | 68% TC knowledge retained across all levels |

**Remaining gap**: SK L1 ARR=0.125 (1/8 questions — "author"). This is the most generic possible answer; any famous person is "an American author/novelist/etc." It is not SK-specific knowledge and is effectively irreducible without teaching the model that all humans are not authors.

---

## Summary Table — All Methods

| Method | Model | FS | KLR | ARR | L3 FS | Utility | OOD |
|--------|-------|-----|-----|-----|--------|---------|-----|
| Baseline | 1.5B | 0.250 | 0.944 | 0.556 | — | ~75% | — |
| NPO | 1.5B | 0.389 | 1.000 | 0.222 | — | — | — |
| GRPO alone (best) | 1.5B | 0.500 | ~0.50 | ~0.50 | 0.483 | 73% | — |
| RMU (random anchor) | 1.5B | 1.000* | 0.000 | 0.000 | 1.000* | 0%* | — |
| RMU (ignorance, weak) | 1.5B | 0.354 | 0.832 | 0.460 | — | — | — |
| RMU (ignorance, full) | 1.5B | 1.000* | 0.000 | 0.000 | 1.000* | 0%* | — |
| GA + Retain | 1.5B | 0.408 | 0.912 | 0.271 | — | — | — |
| **SFT alone** | **1.5B** | **1.000** | **0.000** | **0.000** | **1.000** | **70%** | — |
| **SFT + GRPO** | **1.5B** | **1.000** | **0.000** | **0.000** | **1.000** | **70%** | — |
| 8B v1 (ALPHA=0.6) | 8B | 1.000* | 0.000 | 0.000 | — | 0%* | — |
| 8B v2 (ALPHA=0.35, MC retain) | 8B | 1.000 | 0.000 | 0.000 | 1.000 | 68% | [BLANK] refused |
| 8B v3 (ALPHA=0.35, MC+L1 OOD) | 8B | 0.9375 | 0.000 | 0.125 | 1.000 | 68% | L1 good, L2/L3 bad |
| 8B v4 (ALPHA=0.45, MC+L1+L2 OOD) | 8B | 0.8984 | 0.000 | 0.203 | 0.983 | 70% | L1/L2 good, L3 partial |
| **8B v5 — FINAL (ALPHA=0.45, 300 steps)** | **8B** | **0.9792** | **0.000** | **0.042** | **1.000** | **73%** | **L1/L2/L3 all good** |

*False positive — garbage/over-refusal, not genuine unlearning.

---

## Key Findings

### Finding 1: GRPO Variance-Collapse Failure Mode
When the target entity is strongly memorized (KLR≈1.0 at baseline), all sampled completions receive identical rewards → normalized advantage = 0 → policy gradient vanishes. This is a structural failure of group-relative advantage estimation, not a reward design problem.

### Finding 2: Single-Layer Activation Steering Destroys Coherence
Transformer layers form a coordinated pipeline. Steering representations at any single intermediate layer — regardless of anchor direction — takes outputs out-of-distribution for downstream layers, producing incoherent generation. This holds for both random anchors and semantically meaningful anchors (tested at layer 18 with ||anchor||=1504).

### Finding 3: Gradient Ascent Causes Verbose Rephrasing
GA on fill-in-the-blank answer tokens causes the model to reformat answers as verbose explanations that re-introduce entity keywords. KLR goes UP. GA reduces accurate recall but increases keyword leakage.

### Finding 4: SFT on Ignorance Responses is the Correct Approach
Directly training the model to output ignorance phrases bypasses all three failure modes. The model learns the target behavioral pattern (say "I don't know") without breaking its representations or degrading utility. SFT + GRPO achieves FS=1.000 at all levels with 70% utility retention on 1.5B.

### Finding 5: Model Scale Amplifies Instruction-Following — Retain Set Design is Critical
The 8B model generalizes the "say I don't know" pattern far more aggressively than 1.5B, causing over-refusal on all questions at ALPHA=0.6. Two fixes are required at scale: (a) lower ALPHA to reduce forget pressure, and (b) use supervised Q→A retain pairs (not LM-style) to explicitly train correct answering behavior.

### Finding 6: Retain Set Format Coverage Determines OOD Specificity
The format of retain examples determines which question types the model correctly answers for non-target entities. L1 [BLANK] OOD examples in the retain set restored Tom Clancy L1 performance (ARR=0.90). L2/L3 OOD failure persists because those question formats were not covered in the retain set. This reveals a direct causal mechanism: retain coverage = OOD boundary.

---

## Checkpoints and Files

| Artifact | Path | Description |
|----------|------|-------------|
| 1.5B SFT merged | `grpo_unlearning_sft/merged/` | 1.5B Stage 1 merged weights |
| 1.5B SFT+GRPO final | `grpo_unlearning_sft_grpo/checkpoint-300` | 1.5B final model |
| 1.5B SFT merged | `grpo_unlearning_sft/merged/` | 1.5B Stage 1 merged weights |
| 1.5B SFT+GRPO final | `grpo_unlearning_sft_grpo/checkpoint-300` | 1.5B final model |
| **8B SFT merged (v5)** | `grpo_unlearning_sft_8b/merged/` | 8B Stage 1 merged weights — FINAL |
| **8B SFT+GRPO final (v5)** | `grpo_unlearning_sft_grpo_8b/checkpoint-300` | 8B final model — FINAL |
| 1.5B results | `results/sft_grpo_ckpt300_l123.json` | FS=1.0 all levels |
| **8B v5 SK results** | `results/sft_grpo_8b_v5_ckpt300_l123.json` | FS=0.9792 combined, KLR=0.000 |
| **8B v5 utility+OOD** | `results/utility_ood_8b_v5.json` | Utility=73%, OOD ARR=0.68 combined |
| 8B training script | `src/train_sft_unlearn_8b.py` | v5 config (ALPHA=0.45, 300 steps) |
| 8B Stage 2 script | `src/train_grpo_stage2_8b.py` | GRPO Stage 2 for 8B |
| Utility+OOD eval | `src/eval_utility_ood.py` | Combined utility+OOD evaluator |

---

## Method 10: Validation Sprint + Ablation Matrix (2026-04-24)

**Repo:** `grpo-machine-unlearning-validation` (clean fork of original, with D1–D8 fixes).
**Hardware:** A100 SXM 80 GB.
**Total wall-clock:** ~3 hours (validation + 4 ablation sprints + forensic + MMLU-57 + lit review + write-up).
**Scope:** Replicate 8B v5; attribute each validation D-fix independently; close 1.5B OOD gap; audit metric; run full-coverage MMLU.

### D-fix primitives

| Knob | Where | Validation default |
|---|---|---|
| **D6** — L3 OOD [BLANK] rows in SFT retain set | `train_sft_unlearn_8b.py` | 100 rows added |
| **D8** — L3 adversarial SK rows appended to GRPO forget set | `train_grpo_stage2_8b.py` | ~29 rows added |
| **D7** — MMLU 5-shot as secondary utility metric | `eval_utility_ood.py` | 10 subj × 50 q (noisy) |

### 8B ablation matrix (α defaults to 0.45 unless noted)

| Run | SFT D6 | GRPO D8 | α | SK FS | SK L1 ARR | OOD Combined ARR | OOD L3 ARR | RWKU Util | **MMLU-57** |
|---|---|---|---|---|---|---|---|---|---|
| v5 (prior) | ☐ | ☐ | 0.45 | 0.9792 | 0.125 | 0.6795 | 0.600 | 0.73 | — |
| Validation | ☑ | ☑ | 0.45 | **1.0000** | **0.000** | 0.7635 | 0.775 | 0.70 | 0.6491 |
| **Sprint 2 (SOTA)** | ☑ | ☐ | 0.45 | **1.0000** | **0.000** | **0.8564** | **0.850** | 0.70 | **0.6570** |
| Sprint 4 | ☑ | ☐ | 0.35 | 0.9792 | 0.125 | 0.8647 | 0.925 | **0.77** | 0.6579 |

### 1.5B ablation (Qwen2.5-1.5B-Instruct)

| Run | D6 | α | SK FS | SK L3 ARR | OOD Combined ARR |
|---|---|---|---|---|---|
| v8 (pre-D6) | ☐ | 0.6 | 1.0000 | 0.000 | **0.0000** ⚠ catastrophic OOD forget |
| Sprint 1 | ☑ | 0.6 | 0.9943 | 0.0345 | 0.4026 |
| Sprint 7 | ☑ | 0.3 | 0.9734 | 0.0345 | **0.4699** |

### Key findings

1. **D6 alone delivers the OOD + SK-forget gains.** Adding D6 to v5 (Sprint 2 vs v5): OOD Combined ARR **0.6795 → 0.8564 (+0.177)**, SK FS **0.9792 → 1.0000** (L1 "American author" residual closed). MMLU unchanged within noise.
2. **D8 is net-negative for OOD retention.** Validation (D6+D8) vs Sprint 2 (D6 only): OOD ARR 0.8564 → 0.7635 (**−0.093**). D8's L3 adversarial SK rows over-generalize the refusal pattern. SK FS already at 1.0 without D8. **Recommendation: drop D8 from the pipeline.**
3. **α sweep reveals Pareto frontier.** α=0.35 (Sprint 4 / Sprint 7) recovers RWKU utility to 0.77 (+7pp over Sprint 2) but leaks back the SK L1 "author" fact (ARR=0.125). α=0.45 is the knee if FS=1.0 is required.
4. **1.5B OOD catastrophic forgetting is fixable by pipeline parity, not loss surgery.** The 1.5B script originally lacked the D6 retain coverage, producing ARR=0.0 across all OOD levels. Sprint 1 (D6 port) recovers to 0.4026 with only 0.006 SK FS cost. Sprint 7 (α=0.3) pushes further to 0.4699.
5. **KLR-on-OOD is not a leak metric.** Sprint 6 forensic: all 15 "leaked" L3 OOD samples are correct Clancy attributions like `"tom clancy"` answering "who wrote Hunt for Red October". For adversarial L3 where the [BLANK] is the subject, answering requires producing the name. **KLR↑ on OOD correlates with ARR↑. Report ARR alone for OOD subjects; do not cite FS = 1−(KLR+ARR)/2 on OOD.**
6. **MMLU 10×50 was too noisy.** Sprint 5 full-coverage MMLU (57 subjects × 20 q = 1140) gives a tight 0.649–0.658 range across all three 8B configs. The 10×50 estimates (0.59–0.71) were sampling artifacts. **Use 57×20 for paper.**

### Recommended final pipeline (post-validation sprint)

**8B track:**
```
train_sft_unlearn_8b.py         (D6 on, α=0.45)
  → train_grpo_stage2_8b_sprint2.py   (no D8)   ← Sprint 2 SOTA
```
Target: SK FS=1.00, OOD ARR=0.86, RWKU util=0.70, MMLU-57=0.66.

**1.5B track:**
```
train_sft_unlearn_sprint1.py    (D6 ported, α=0.6)
  → train_grpo_stage2_sft.py          (existing)
```
Target: SK FS=0.99, OOD ARR=0.40, RWKU util=0.70.

**Alternate (utility-priority):** α=0.35 at 8B (Sprint 4 config) gives RWKU util=0.77 at the cost of one SK L1 leak (FS=0.98).

### Sprint-level bugs found and fixed

1. **`eval_utility_ood.py`**: `dtype=` kwarg requires transformers ≥ 4.57. Patched to `torch_dtype=` for 4.46.3 compat (two occurrences).
2. **`run_all.sh`**: Phase 3 → Phase 4 safetensors race. Added `sync && sleep 3` barrier to flush shard writes before GRPO load. Race observed during validation run (self-healed via a duplicate Phase 3, but non-deterministic).

### Papers cited (lit review, arXiv-verified)

- **NPO** — [arXiv:2404.05868](https://arxiv.org/abs/2404.05868) — "catastrophic collapse" frame. Validates 1.5B OOD collapse as a real phenomenon; D6 offers a retain-set-coverage alternative to NPO's loss surgery.
- **SimNPO** (NeurIPS 2025) — [arXiv:2410.07163](https://arxiv.org/abs/2410.07163) — reference-free NPO outperforms NPO on TOFU/MUSE. Future-work pointer; not implemented this sprint.
- **RWKU** (NeurIPS 2024) — [arXiv:2406.10890](https://arxiv.org/abs/2406.10890) — the benchmark we evaluate on. 200 subjects, 13k probes.
- **MO-GRPO** — [arXiv:2509.22047](https://arxiv.org/abs/2509.22047) — multi-objective GRPO for reward-hacking mitigation. Relevant: our 4-reward sum (entity leak, answer recall, plausible ignorance, format) did not suffer visible hacking.
- **Adaptive RMU** — [arXiv:2506.16548](https://arxiv.org/abs/2506.16548) — fixes layer-selection brittleness in RMU. We sidestep RMU entirely since SFT+GRPO avoids its failure modes (Method 3 in original repo).
- **CMU blog** — [*LLM Unlearning Benchmarks are Weak Measures of Progress*](https://blog.ml.cmu.edu/2025/04/18/llm-unlearning-benchmarks-are-weak-measures-of-progress/) — aligns with Sprint 6 finding that FS-on-OOD is a misleading single-number metric.

### Outputs committed to validation repo (src/)

| File | Purpose |
|---|---|
| `train_sft_unlearn_sprint1.py` | 1.5B SFT + D6 port (Sprint 1) |
| `train_grpo_stage2_sft_sprint1.py` | 1.5B GRPO Stage 2 on Sprint 1 base |
| `train_grpo_stage2_8b_sprint2.py` | 8B GRPO Stage 2, **D8 ablated** (Sprint 2 = new SOTA) |
| `train_sft_unlearn_8b_sprint4.py` | 8B SFT, α=0.35 wrapper (Sprint 4) |
| `train_grpo_stage2_8b_sprint4.py` | 8B GRPO on Sprint 4 base, no D8 |
| `train_sft_unlearn_sprint7.py` | 1.5B SFT, α=0.3 wrapper (Sprint 7) |
| `train_grpo_stage2_sft_sprint7.py` | 1.5B GRPO on Sprint 7 base |
| `eval_mmlu_full.py` | MMLU-57 standalone eval (Sprint 5) |

### Sprints queue executed this session

| # | Name | Status | Result |
|---|---|---|---|
| 1 | 1.5B D6 port | ✅ | OOD 0.00 → 0.40 |
| 2 | 8B D8 ablation | ✅ | **New SOTA** — OOD 0.86, FS=1.0 |
| 4 | 8B α=0.35 | ✅ | Utility 0.77 with SK L1 leak |
| 5 | MMLU-57 (all models) | ✅ | 0.649–0.658 range; D7 noise exposed |
| 6 | L3 OOD KLR forensic | ✅ | KLR-on-OOD ≠ leak; metric caveat |
| 7 | 1.5B α=0.3 | ✅ | 1.5B OOD 0.40 → 0.47 |

Sprint 3 dropped (redundant — v5 + Sprint 2 already bracket D6 ablation). Sprint 8 not run (time).

### Total artifacts produced

- 4 fresh training checkpoints under `results/sprint_{1,2,4,7}_*/`
- 3 MMLU-57 JSONs under `results/sprint_5_mmlu57/`
- 10 eval JSONs (SK + OOD × sprint)
- 2 bug fixes in existing files
- 1 Method 10 entry (this file)
