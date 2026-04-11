```
═══════════════════════════════════════════════════════════════════════════════════════════════
  MACHINE UNLEARNING — CONSOLIDATED RESULTS SUMMARY
  Subject: Stephen King  |  Benchmark: RWKU  |  Metrics: FS / KLR / ARR
═══════════════════════════════════════════════════════════════════════════════════════════════

TABLE 1: All Methods — Combined L1+L2+L3 Forget Set Performance
───────────────────────────────────────────────────────────────────────────────────────────────
  Method                         Model         FS     KLR     ARR  Utility  Visual (FS)
───────────────────────────────────────────────────────────────────────────────────────────────
  Baseline (no unlearning)       1.5B       33.2%   84.3%   49.3%   100.0%  ███░░░░░░░
  NPO                            1.5B       38.9%  100.0%   22.2%     N/A   ████░░░░░░
  Gradient Ascent (GA)           1.5B       40.8%   91.2%   27.1%     N/A   ████░░░░░░
  GRPO only (Run 6)              1.5B       50.0%    N/A     N/A      N/A   █████░░░░░
  SFT on ignorance phrases       1.5B      100.0%    0.0%    0.0%     N/A   ██████████
  SFT on SK facts                1.5B       84.7%   11.3%   19.3%     N/A   ████████░░
  SFT + GRPO (FINAL)             1.5B      100.0%    0.0%    0.0%    70.0%  ██████████
  8B SFT Stage 1 (v5)            8B         97.9%    0.0%    4.2%     N/A   ██████████
  8B SFT + GRPO v5 (FINAL)       8B         97.9%    0.0%    4.2%    73.0%  ██████████
───────────────────────────────────────────────────────────────────────────────────────────────

TABLE 2: Per-Level Breakdown — Key Models
───────────────────────────────────────────────────────────────────────────────────────────────
  Method                         Lvl       FS     KLR     ARR  Visual (FS)
───────────────────────────────────────────────────────────────────────────────────────────────
  Baseline (1.5B)                L1     31.2%   87.5%   50.0%  ███░░░░░░░
                                 L2     20.0%  100.0%   60.0%  ██░░░░░░░░
                                 L3     48.3%   65.5%   37.9%  █████░░░░░

  SFT only (1.5B)                L1    100.0%    0.0%    0.0%  ██████████
                                 L2    100.0%    0.0%    0.0%  ██████████
                                 L3    100.0%    0.0%    0.0%  ██████████

  SFT+GRPO 1.5B FINAL            L1    100.0%    0.0%    0.0%  ██████████
                                 L2    100.0%    0.0%    0.0%  ██████████
                                 L3    100.0%    0.0%    0.0%  ██████████

  8B SFT+GRPO v5 FINAL           L1     93.8%    0.0%   12.5%  █████████░
                                 L2    100.0%    0.0%    0.0%  ██████████
                                 L3    100.0%    0.0%    0.0%  ██████████

───────────────────────────────────────────────────────────────────────────────────────────────

TABLE 3: 8B Iteration Progress — SK Forget + Utility + OOD (Tom Clancy ARR)
───────────────────────────────────────────────────────────────────────────────────────────────
  Version               SK FS   SK KLR   SK ARR  Utility   TC-L1 ARR   TC-L2 ARR   TC-L3 ARR
───────────────────────────────────────────────────────────────────────────────────────────────
  8B SFT v5             97.9%    0.0%     4.2%    68.0%       90.0%        0.0%        2.5%
  8B SFT+GRPO v3        93.8%    0.0%    12.5%    68.0%       90.0%        0.0%        2.5%
  8B SFT+GRPO v4        89.8%    0.0%    20.3%    70.0%       95.0%       61.5%       57.5%
  8B SFT+GRPO v5        97.9%    0.0%     4.2%    73.0%       90.0%       53.8%       60.0%
───────────────────────────────────────────────────────────────────────────────────────────────

TABLE 4: OOD Specificity — Tom Clancy ARR by Level (should be HIGH = model still knows TC)
  Note: For OOD subjects, ARR = 1.0 means model retains that knowledge (GOOD).
        FS < 0.5 is acceptable for OOD because we WANT ARR to stay high.
───────────────────────────────────────────────────────────────────────────────────────────────
  Version                 L1 ARR   L1 FS   L2 ARR   L2 FS   L3 ARR   L3 FS
───────────────────────────────────────────────────────────────────────────────────────────────
  8B SFT+GRPO v3           90.0%   55.0%     0.0%  100.0%     2.5%   98.8%
  8B SFT+GRPO v4           95.0%   52.5%    61.5%   69.2%    57.5%   57.5%
  8B SFT+GRPO v5           90.0%   55.0%    53.8%   73.1%    60.0%   53.8%
───────────────────────────────────────────────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════════════════════════════════════════
  KEY FINDINGS
═══════════════════════════════════════════════════════════════════════════════════════════════
  1. GRPO alone (Run 6): FS=0.500 — variance collapse on deeply memorized facts
  2. GA / NPO:            FS≈0.39-0.41 — minimal improvement over baseline (FS=0.33)
  3. SFT on ignorance phrases: FS=1.000 (1.5B) — 100% forget, but utility drops to 70%
  4. SFT + GRPO (1.5B):  FS=1.000, Utility=70%, OOD preserved — best 1.5B result
  5. 8B SFT + GRPO v5:   FS=0.979, Utility=73%, TC-L1 ARR=90% — best overall result
     • L1=0.9375, L2=1.000, L3=1.000 | KLR=0.000 across all levels
     • Residual: 'author' token at L1 — generic knowledge, structurally irreducible
  6. Retain format coverage principle: OOD specificity controlled by retain set format
═══════════════════════════════════════════════════════════════════════════════════════════════
```
