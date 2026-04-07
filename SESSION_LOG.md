# Project Session Log — GRPO Machine Unlearning

**Model:** Qwen2.5-1.5B-Instruct  
**Dataset:** RWKU (jinzhuoran/RWKU) — Stephen King (primary), Tom Clancy, Leonardo da Vinci (OOD)  
**Hardware:** NVIDIA A40, 47.7GB VRAM  
**Course:** USF CIS 4930/6930 Deep RL  
**Student:** Nithin2311 (palyam@usf.edu)

---

## Metrics Legend

- **FS** = Forget Score = 1 − (KLR + ARR) / 2 — **higher = better unlearning**
- **KLR** = Keyword Leak Rate — fraction of responses mentioning the entity name (lower = better)
- **ARR** = Answer Recall Rate — fraction of responses containing the correct answer (lower = better)
- **US** = Utility Score — general knowledge retention on RWKU utility_general MC (higher = better)
- **L1** = RWKU direct fill-in-blank | **L2** = harder direct | **L3** = indirect/paraphrase
- **Quick eval** = L1+L2 only, n=18 | **Full eval** = L1+L2+L3 combined

---

## Baselines (Qwen2.5-1.5B-Instruct, no training)

| Subject | Eval Type | FS | KLR | ARR | US |
|---------|-----------|----|----|-----|----|
| Stephen King | Quick (L1+L2, n=18) | 0.250 | 0.944 | 0.556 | — |
| Stephen King | Full (L1+L2+L3) | 0.332 | 0.843 | 0.493 | 1.000 |
| Tom Clancy | Full (L1+L2+L3) | 0.484 | 0.587 | 0.444 | — |
| Leonardo da Vinci | Full (L1+L2+L3) | 0.204 | — | — | — |
| ↳ da Vinci L1 | — | 0.350 | 0.600 | 0.700 | — |
| ↳ da Vinci L2 | — | 0.088 | **1.000** | 0.824 | — |
| ↳ da Vinci L3 | — | 0.175 | 0.950 | 0.700 | — |

**Key observation:** Da Vinci L2 KLR=1.000 — the model outputs the entity name on 100% of L2 questions at baseline. This is the root cause of GRPO failure on da Vinci.

---

## Method 1: GDA — Gradient Descent Ascent (Stephen King)

**Approach:** Maximize cross-entropy loss on forget set (push model away from correct answers).  
**Rationale:** Simplest parametric unlearning — direct weight-level intervention, no reward shaping needed.  
**Config:** Standard GDA, no retain loss, LoRA fine-tuning.

| Checkpoint | FS | KLR | ARR |
|------------|----|-----|-----|
| GDA ckpt100 | 0.333 | 0.889 | 0.444 |
| GDA ckpt300 | 0.389 | 0.833 | 0.389 |
| GDA final   | 0.444 | 0.778 | 0.333 |

**Finding:** Reduces KLR and ARR consistently. FS improves to 0.444 (baseline 0.250). No utility measurement taken — likely degrades general capability. No L3 robustness tested.

---

## Method 2: GRPO — Group Relative Policy Optimization (Stephen King, Runs 1–6)

**Approach:** RL with 4 reward functions:
1. `entity_leak_penalty_reward`: −2.0 if entity name appears, +0.5 if clean
2. `answer_recall_penalty_reward`: −3.0 if ground-truth answer appears, +0.5 if absent
3. `plausible_ignorance_reward`: up to +4.0 for "I don't know" phrasing with no entity leak
4. `format_adherence_reward`: up to +0.4 for fluency, −1.0 for repetitive/short output

**Evolution across runs:**

| Run | Key Change | FS (quick) | Notes |
|-----|-----------|-----------|-------|
| Runs 1–3 | Initial prototype, reward tuning, QLoRA | ~0.200–0.250 | Baseline GRPO iteration |
| Run 4 ckpt400 | Strengthened reward weights | 0.306 | Marginal gain |
| Run 5 ckpt100 | Augmented data 3× | 0.278 | Diminishing returns |
| **Run 6 ckpt300** | **Augmented data 6× (280 samples), lr=2e-6** | **0.500 (full eval)** | **WINNER** |

**Run 6 winner config:**
- lr=2e-6, beta=0.1 (KL penalty), 500 steps, LoRA r=16, all projections (q/k/v/o/gate/up/down)
- Forget set: 280 augmented samples (6 paraphrases × 47 original questions)
- bf16, no quantization — Utility score: **0.730**

**L3 Robustness test on Run6/ckpt300:**

| Level | Baseline FS | Trained FS | Delta |
|-------|------------|-----------|-------|
| L1 (direct) | 0.313 | 0.375 | +0.063 |
| L2 (harder direct) | 0.200 | 0.200 | **0.000** |
| L3 (paraphrase) | 0.483 | 0.483 | **0.000** |

**Critical finding:** GRPO does NOT survive paraphrase-based probing. L2 and L3 FS unchanged. Model learns shallow surface patterns (token-level keyword avoidance) but underlying parametric knowledge is intact. This is the core limitation of output-level reward shaping.

---

## Method 3: NPO — Negative Preference Optimization (Stephen King)

**Approach:** DPO-style: treat forget-set responses as "rejected." Loss = `log(1 + exp(β × log_ratio))`.  
**Config:** lr=2e-5, beta=0.1, alpha=0.5, 500 steps, LoRA r=32, retain n=200.

| Checkpoint | FS | KLR | ARR |
|------------|----|-----|-----|
| NPO ckpt100 | 0.333 | 0.944 | 0.389 |
| NPO ckpt200 | 0.361 | 1.000 | 0.278 |
| NPO ckpt400 | 0.417 | 1.000 | 0.167 |
| NPO final   | 0.389 | 1.000 | 0.222 |

**Finding:** Achieves KLR=1.000 (keyword suppression), but over-suppresses — model becomes evasive globally. ARR=0.167 at ckpt400 is good but FS plateau at 0.417 shows utility trade-off. Utility likely severely degraded (not measured).

---

## Method 4: SFT Refusal (Stephen King)

**Approach:** Supervised fine-tuning to produce refusal responses for all forget-set questions.  
**Config v1 (collapsed):** lr=2e-4, retain = RWKU MC format (format mismatch with forget set Q&A).  
**Config v2 (fixed):** lr=5e-5, retain = RWKU open-ended Q&A for other subjects (same format), retain weight=2.0, batch=16, FA2, bf16.

| Checkpoint | FS (L1+L2+L3) | KLR | ARR | Notes |
|------------|--------------|-----|-----|-------|
| SFT ckpt100 | 0.828 | 0.081 | 0.264 | v2 |
| SFT ckpt200 | 0.847 | 0.113 | 0.193 | v2 |
| SFT ckpt300 | **1.000** | **0.000** | **0.000** | v2 — likely over-refusal |
| SFT final  | 0.847 | 0.113 | 0.193 | v2 |

**Finding:** SFT ckpt300 achieves FS=1.000 — but almost certainly due to catastrophic over-refusal (model refuses everything). Utility not measured. This is behavior suppression, not genuine parametric unlearning. Best practical checkpoint is likely ckpt200 (FS=0.847 with partial leakage vs ckpt300 over-refusal).

---

## Method 5: GRPO — Tom Clancy (simpler entity experiment)

**Hypothesis:** Lower baseline KLR (0.587 vs 0.843) → more output variance → GRPO can learn from it.  
**Config:** Same as Run 6 winner. Subject = Tom Clancy.

| Checkpoint | FS (L1+L2+L3) | KLR | ARR |
|------------|----------------|-----|-----|
| TC baseline | 0.484 | 0.587 | 0.444 |
| TC ckpt100 | 0.480 | 0.605 | 0.436 |
| TC ckpt300 | 0.493 | 0.579 | 0.436 |
| TC ckpt400 | **0.514** | **0.571** | **0.403** |
| TC ckpt500 | 0.488 | 0.596 | 0.428 |

**OOD transfer — SK-trained model on Tom Clancy:**

| | FS | KLR | ARR |
|-|----|-----|-----|
| TC baseline | 0.484 | 0.587 | 0.444 |
| SK model on TC (OOD) | 0.476 | 0.613 | 0.436 |

**Finding:** Even on a less-famous subject, GRPO only marginally improves FS (0.484 → 0.514). OOD shows near-zero transfer — unlearning is subject-specific. The reward signal is simply too weak regardless of entity fame level.

---

## Method 6: GRPO — Leonardo da Vinci (Run 7)

**Hypothesis:** Wider forget set coverage (L1+L2+L3 = 482 samples), extended entity keywords (Mona Lisa, Last Supper, etc.), higher beta/temp will improve results.  
**Config:** lr=2e-6, beta=0.15, temp=1.1, LoRA r=16, 500 steps.

| Checkpoint | Combined FS | L1 FS | L2 FS | L3 FS |
|------------|------------|-------|-------|-------|
| Baseline | 0.204 | 0.350 | **0.088** | 0.175 |
| ckpt100 | 0.225 | 0.350 | **0.088** | 0.238 |
| ckpt300 | 0.225 | 0.375 | **0.088** | 0.213 |
| ckpt500 | 0.217 | 0.350 | **0.088** | 0.213 |

**Finding:** GRPO completely fails on da Vinci. L2 FS stuck at 0.088 across all checkpoints — identical to baseline. Root cause: baseline L2 KLR=1.000 means every output leaks the name → no variance → no GRPO gradient signal. Confirms variance-collapse failure mode is the fundamental bottleneck.

---

## Summary: All Methods Compared

| Method | Subject | Best FS | KLR | ARR | L3 Robust? | Utility |
|--------|---------|---------|-----|-----|------------|---------|
| Baseline | Stephen King | 0.250 | 0.944 | 0.556 | — | 1.000 |
| GDA | Stephen King | 0.444 | 0.778 | 0.333 | Not tested | Unknown |
| **GRPO Run6** | **Stephen King** | **0.500** | **0.556** | **0.444** | **NO** | **0.730** |
| NPO | Stephen King | 0.417 | 1.000 | 0.167 | Not tested | Unknown |
| SFT Refusal v2 | Stephen King | 1.000* | 0.000 | 0.000 | Not tested | ~0 (over-refusal) |
| GRPO | Tom Clancy | 0.514 | 0.571 | 0.403 | Not tested | Unknown |
| GRPO Run7 | Leonardo da Vinci | 0.225 | — | — | NO | Unknown |

*SFT FS=1.000 at ckpt300 is likely catastrophic over-refusal, not genuine unlearning.

---

## Core Problem Identified: GRPO Variance-Collapse

GRPO computes group-relative rewards within a batch. If all completions in the batch get the same reward (e.g., all get −2.0 for entity leak), the normalized advantage is zero → zero gradient → no learning. This happens when:
- Baseline KLR is near 1.0 (deeply embedded subjects): every completion leaks the name
- Model converges on a degenerate response: e.g., always refuses in the same wording

Even when GRPO succeeds (Stephen King), it only modifies token-level output patterns. The parametric memory (weights encoding the facts) is unchanged — paraphrase probing reveals this (L3 FS unchanged at 0.483).

---

## Planned Next Step: Two-Stage RMU → GRPO

### What is RMU (Representation Misdirection Unlearning)
RMU operates at the internal representation level rather than the output level:
- For forget-set inputs: steer hidden states at a target layer toward a random noise vector (disrupts the encoding that retrieves factual knowledge)
- Simultaneously: keep retain-set hidden states stable (preserves utility)
- This creates a direct parametric intervention — does not require output variance

### Why RMU + GRPO together
- **RMU alone**: Disrupts knowledge but may produce incoherent outputs
- **GRPO alone**: Fails when KLR≈1.0 at baseline (no variance)
- **RMU → GRPO**: RMU first breaks the knowledge substrate (now KLR drops, creating variance), then GRPO shapes the outputs toward plausible ignorance. Synergistic.

### Academic framing for the report
"We identify GRPO's variance-collapse failure mode empirically across three subjects. We propose a two-stage pipeline: Stage 1 (RMU) disrupts parametric memory at the representation level; Stage 2 (GRPO) refines surface behavior using reward shaping. GRPO remains the primary optimization framework."

### Subject: Need less embedded celebrity
All tested subjects are too famous for GRPO alone. Target: RWKU subject with baseline KLR ≈ 0.45–0.65. Candidates: Jeff Goldblum, Kiefer Sutherland, Brett Favre. Must verify RWKU membership and run quick baseline probe before training.

### Implementation plan
- `src/train_rmu_stage1.py`: ~100 lines — hook layer 18, push forget hidden states → random vector, retain loss
- `src/train_grpo_stage2_rmu.py`: minor modification of train_grpo_davinci.py, load from RMU checkpoint
- Expected runtime: ~70 min total (30 min RMU + 40 min GRPO) on A40

---

## Key Decisions & Pivots

| Decision | Reason |
|----------|--------|
| Dropped Unsloth | Incompatible with PyTorch 2.4.1 (needs 2.5+) |
| Dropped 4-bit QLoRA for 1.5B | Unnecessary on A40 (48GB); 1.5B needs ~6GB in bf16 |
| Switched TRL 1.0.0 → 0.15.2 | TRL 1.0.0 imports FSDPModule requiring PyTorch 2.5+ |
| Kept 4 reward functions | Single reward produces degenerate outputs; multi-reward balances leak/fluency/ignorance |
| Raised beta 0.1 → 0.15 (Run7) | More KL penalty to preserve utility — no measurable effect on da Vinci |
| Raised temp 0.9 → 1.1 (Run7) | Increase output variance — still insufficient for KLR=1.0 baseline |
| Dropped DPO standalone | Project is GRPO RL unlearning; DPO is offline, no RL component |

---

## Deadlines

| Date | Milestone |
|------|-----------|
| April 10, 2026 | Personal checkpoint (NOT a course deadline) |
| April 24, 2026 | 8B scaling (Llama-3.1-8B, L3 + OOD) — **hard deadline** |
| May 1, 2026 | Presentation |
| May 7, 2026 | Final report |

---

## File Map

| File | Purpose |
|------|---------|
| `src/train_grpo_davinci.py` | Run 7 training (Leonardo da Vinci) |
| `src/train_grpo_tom_clancy.py` | Tom Clancy GRPO training |
| `src/train_npo.py` | NPO training |
| `src/train_sft_refusal.py` | SFT refusal training |
| `src/eval_multilevel.py` | L1+L2+L3 evaluation |
| `src/eval_quick.py` | Fast L1+L2 evaluation |
| `src/reward_functions.py` | All 4 GRPO reward functions |
| `src/augment_davinci.py` | Da Vinci data augmentation |
| `src/augment_tom_clancy.py` | Tom Clancy data augmentation |
| `data/stephen_king_augmented.json` | SK forget set (280 samples, 6× aug) |
| `data/tom_clancy_augmented.json` | TC forget set |
| `data/leonardo_augmented.json` | Da Vinci forget set (482 samples, L1+L2+L3) |
| `results/davinci_eval_progress.json` | Da Vinci L1+L2+L3 across all checkpoints |
| `results/tc_grpo_*.json` | Tom Clancy GRPO eval results |
| `results/tc_ood_run6ckpt300.json` | OOD: SK model evaluated on TC |
| `results/sft_sk_*.json` | SFT refusal results |
| `results/npo_ckpt*.json` | NPO checkpoint results |
| `results/full_eval_run6_ckpt300.json` | SK Run6 winner full eval with utility |
| `grpo_unlearning_davinci/` | Run 7 checkpoints (ckpt100–500) |
| `grpo_unlearning_tom_clancy/` | TC GRPO checkpoints |
| `grpo_unlearning_npo/` | NPO checkpoints |
| `grpo_unlearning_sft_sk/` | SFT refusal checkpoints |
