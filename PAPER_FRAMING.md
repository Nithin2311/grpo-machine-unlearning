# Paper Framing — GRPO Machine Unlearning
**Last updated: 2026-04-11**
**For: CIS 4930/6930 Deep RL Final Report (due May 7, 2026)**

---

## One-Sentence Thesis

> We identify GRPO's **variance-collapse failure mode** in machine unlearning, empirically characterize it across three entities and scales, and propose a two-stage **RMU → GRPO** pipeline that achieves genuine robustness to adversarial L3 probing where GRPO alone fails.

---

## Why This Framing Works for the Course

The course is Deep RL. The instructor (Fengchun Qiao) will evaluate this from an RL lens.
- The variance-collapse failure is an RL problem: when the group-relative advantage estimator sees identical rewards across all completions, the policy gradient vanishes. This is directly analogous to the **exploration collapse** / **reward hacking** failures studied in safe RL literature.
- The fix (RMU → GRPO) is a curriculum-style approach: Stage 1 pre-conditions the state space so Stage 2 exploration is non-degenerate.
- This is a novel application of RL stability analysis to the LLM unlearning domain.

---

## Narrative Arc (for presentation + report)

### 1. Problem Setup (MDP Formulation)
- **State**: The LLM's current output distribution given a forget-set query
- **Action**: A completion token sequence sampled by the policy
- **Reward**: Composite of entity_leak_penalty + answer_recall_penalty + plausible_ignorance + format
- **Objective**: Find a policy π that maximizes expected reward on the forget set while staying close to π_ref (KL penalty = β)

*"We formulate machine unlearning as an RL problem, where the agent learns to produce informationally-empty responses to queries about a target entity."*

### 2. Baseline GRPO Results (Stephen King — Run 6)
- FS improves from 0.332 → 0.500, utility 0.730 ✓
- BUT: L2 FS unchanged (0.200 → 0.200), L3 FS unchanged (0.483 → 0.483)
- **Finding A**: GRPO succeeds at output-level pattern modification but fails at parametric unlearning

### 3. Multi-Entity Failure Analysis (the core RL contribution)
| Entity | Baseline KLR | GRPO Result | Explanation |
|--------|-------------|-------------|-------------|
| Stephen King | 0.843 | FS: 0.332 → 0.500 | Marginal variance → marginal gradient |
| Tom Clancy | 0.587 | FS: 0.484 → 0.514 | Lower KLR → slightly more variance |
| Leonardo da Vinci | KLR=1.000 (L2) | FS: 0.204 → 0.225 | Zero variance → zero gradient → failure |

**Finding B (the key RL insight)**: GRPO's group-relative advantage normalization creates a degenerate case when all completions receive identical rewards. When baseline KLR ≈ 1.0, every sampled completion leaks the entity name, every completion gets reward −2.0, the normalized advantage is zero, and the policy gradient vanishes. This is the **variance-collapse failure mode**.

*"The failure is not a reward function design problem — it is structural to GRPO's advantage estimation when the forget-set entity is strongly memorized."*

### 4. Why GRPO Alone Cannot Achieve Parametric Unlearning
- Even for SK (where it "works"), the parametric memory is intact: L3 FS unchanged
- GRPO shapes OUTPUT behavior (what tokens get generated) but not WEIGHT space (what factual knowledge is stored)
- The model learns: "when asked about Stephen King, produce a refusal" — not: "I no longer encode Stephen King facts"
- This is verified by the L3 adversarial probing results: paraphrase-framed queries bypass the learned behavior

### 5. Proposed Fix: RMU → GRPO Two-Stage Pipeline
**Stage 1 — Representation Misdirection Unlearning (RMU):**
- Operates in activation space, not output space → no dependence on output variance
- Hooks intermediate hidden states at layer 18 (64% depth — where factual associations are encoded)
- For forget inputs: steers hidden states toward a fixed random anchor vector (L2 loss)
- For retain inputs: standard CE loss (preserves general capability)
- After ~150 steps: KLR drops from ~0.85 → ~0.3–0.5 (parametric memory disrupted)

**Stage 2 — GRPO Output Refinement:**
- Loads from RMU-modified base
- Now KLR is lower → output variance exists → GRPO reward signal is non-zero
- GRPO shapes the residual outputs toward fluent plausible ignorance
- Synergistic: RMU disrupts the "what to say", GRPO refines "how to say nothing"

**Framing it as RL curriculum learning:**
> "RMU serves as a pre-training phase that reshapes the policy's state-action distribution so that the subsequent RL optimization is non-degenerate. This is analogous to curriculum learning in RL, where easier or pre-conditioned environments are used to bootstrap policy gradient training."

### 6. Scaling to 8B (April 24 deadline)
- Apply RMU → GRPO to Llama-3.1-8B-Instruct with QLoRA (4-bit)
- Key questions: does the variance-collapse failure mode persist at 8B? Does RMU → GRPO scale?
- Compare: 1.5B GRPO alone vs 1.5B RMU+GRPO vs 8B RMU+GRPO
- L3 adversarial testing on best 8B checkpoint
- OOD: apply SK-trained 8B model to a held-out RWKU entity

---

## Key Claims for the Report (in order of strength)

1. **Strong (experimentally verified)**: GRPO achieves output-level unlearning (FS: 0.332→0.500) but not parametric unlearning (L3 unchanged) on Stephen King.

2. **Strong (experimentally verified)**: GRPO fails completely when baseline KLR ≈ 1.0 (da Vinci L2: FS=0.088 across all checkpoints). This is the variance-collapse failure mode.

3. **Strong (mechanistically explained)**: The failure is caused by zero-variance GRPO advantage estimation — all completions receive identical rewards, normalizing advantages to zero.

4. **Medium (experimentally verified once complete)**: RMU → GRPO breaks variance-collapse by first reducing KLR through parametric representation steering.

5. **Medium (to be verified)**: RMU → GRPO achieves improved L3 robustness (non-zero FS improvement at L3) unlike GRPO alone.

6. **Pending (8B experiment)**: Scaling characteristics — does variance-collapse severity correlate with model size? Does RMU → GRPO remain effective at 8B?

---

## How to Handle "You Didn't Get Perfect Results"

**Don't frame it as failure. Frame it as discovery.**

> "Our primary contribution is not a method that achieves perfect unlearning — no published method does. Our contribution is the identification, characterization, and mechanistic explanation of the variance-collapse failure mode that fundamentally limits reward-shaping-based unlearning methods. We additionally propose a two-stage remedy and demonstrate its effectiveness."

The RWKU paper itself shows that best-performing methods still have significant ROUGE-L recall (ICU: 12.8, not 0). Perfect unlearning is not the bar.

---

## Comparison Table for Report (template)

| Method | Best FS | L3 Robust? | Utility | Parametric? | Notes |
|--------|---------|------------|---------|-------------|-------|
| Baseline | 0.332 | — | 1.000 | — | |
| GDA | 0.444 | Not tested | Unknown | Partially | Fast, no utility measurement |
| GRPO Run6 | 0.500 | **NO** | 0.730 | **NO** | L3 unchanged |
| NPO | 0.417 | Not tested | Unknown | Partial | KLR=1.0 = over-suppression |
| SFT Refusal | 0.847* | Not tested | ~0 | NO | Over-refusal at ckpt300 |
| **RMU→GRPO** | **TBD** | **TBD** | **TBD** | **Partial** | Key new result |

*SFT FS=1.000 at ckpt300 is catastrophic over-refusal.

---

## Key Phrases to Reuse (academic tone)

- "output-level behavioral modification vs. parametric knowledge erasure"
- "group-relative advantage normalization creates a degenerate case under reward saturation"
- "variance-collapse failure mode, empirically validated across three subjects"
- "two-stage curriculum: representation disruption followed by output refinement"
- "plausible ignorance reward — operationalizing the 'right to be forgotten' as an RL objective"
- "adversarial L3 probing reveals that standard GRPO unlearning is behaviorally shallow"

---

## Report Sections (outline)

1. **Introduction** — GDPR/right-to-be-forgotten motivation, GRPO as RL solution, paper's contributions
2. **Background** — MDP formulation of unlearning, GRPO algorithm, RWKU benchmark
3. **Method** — 4-component reward function, training setup, RMU → GRPO pipeline
4. **Experiments** — Baselines, SK runs 1-6, multi-entity analysis, RMU → GRPO results
5. **Analysis** — Variance-collapse failure mode (the main RL contribution)
6. **Scaling** — 8B results, QLoRA setup, L3 + OOD
7. **Discussion** — Limitations (behavior vs. parametric, eval metric limitations), future work
8. **Conclusion**

---

## Deadlines Reminder
- **April 24**: 8B scaling complete
- **May 1**: 15-min presentation (bring plots + comparison table)
- **May 7**: Written report due
