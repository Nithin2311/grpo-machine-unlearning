#!/usr/bin/env python3.11
"""
Reads all key result JSONs and prints a clean comparison table.
Usage: python3.11 src/summarize_results.py [--save]
"""

import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
SAVE_MD = "--save" in sys.argv

# ─── helpers ──────────────────────────────────────────────────────────────────

def load(path):
    p = RESULTS_DIR / path
    if not p.exists():
        return None
    return json.load(open(p))

def fs(d, key="forget_score"):
    """Pull forget_score from various JSON shapes."""
    if d is None:
        return None
    # top-level
    if key in d:
        return d[key]
    # nested under 'combined'
    if "combined" in d and key in d["combined"]:
        return d["combined"][key]
    return None

def klr(d):
    return fs(d, "keyword_leak_rate")

def arr(d):
    return fs(d, "answer_recall_rate")

def util(d):
    if d is None:
        return None
    if "utility_score" in d:
        return d["utility_score"]
    if "utility" in d and isinstance(d["utility"], dict):
        return d["utility"].get("utility_score")
    if "utility" in d and isinstance(d["utility"], (float, int)):
        return d["utility"]
    return None

def level(d, lvl):
    """Return (fs, klr, arr) for a given level string 'L1' etc."""
    if d is None:
        return None, None, None
    levels = d.get("by_level", {})
    if lvl not in levels:
        return None, None, None
    l = levels[lvl]
    return l.get("forget_score"), l.get("keyword_leak_rate"), l.get("answer_recall_rate")

def fmt(v, pct=True):
    if v is None:
        return "  N/A  "
    if pct:
        return f"{v*100:5.1f}%"
    return f"{v:.4f}"

def bar(v, width=10):
    if v is None:
        return " " * width
    filled = int(round(v * width))
    return "█" * filled + "░" * (width - filled)

# ─── load data ────────────────────────────────────────────────────────────────

baseline_1b    = load("sk_l123_baseline.json")
npo            = load("npo_final.json")
ga             = load("ga_merged_l123.json")
grpo_run6      = load("full_eval_run6_ckpt300.json")   # pure GRPO run 6
sft_only       = load("sft_merged_l123.json")          # SFT-only unlearn
sft_sk         = load("sft_sk_final.json")             # SFT on SK facts
sft_grpo_1b    = load("sft_grpo_ckpt300_l123.json")   # 1.5B SFT+GRPO FINAL

sft_8b_v5      = load("sft_8b_v5_merged_l123.json")
sft_grpo_8b    = load("sft_grpo_8b_v5_ckpt300_l123.json")

util_8b_v3     = load("utility_ood_8b_v3.json")
util_8b_v4     = load("utility_ood_8b_v4.json")
util_8b_v5     = load("utility_ood_8b_v5.json")

baseline_full  = load("full_eval_baseline.json")  # 1.5B baseline utility

# ─── OOD data (Tom Clancy) from v3/v4/v5 logs ─────────────────────────────────
# These come from the ood sub-dict of utility_ood files
def ood_levels(d):
    if d is None or "ood" not in d:
        return {}
    return d["ood"].get("by_level", {})

tc_v3 = ood_levels(util_8b_v3)
tc_v4 = ood_levels(util_8b_v4)
tc_v5 = ood_levels(util_8b_v5)

# ─── print ────────────────────────────────────────────────────────────────────

SEP  = "─" * 95
SEP2 = "═" * 95

lines = []
def p(s=""):
    print(s)
    lines.append(s)

p(SEP2)
p("  MACHINE UNLEARNING — CONSOLIDATED RESULTS SUMMARY")
p("  Subject: Stephen King  |  Benchmark: RWKU  |  Metrics: FS / KLR / ARR")
p(SEP2)
p()

# ── TABLE 1: All methods, combined L1+L2+L3 ──────────────────────────────────
p("TABLE 1: All Methods — Combined L1+L2+L3 Forget Set Performance")
p(SEP)
p(f"  {'Method':<30} {'Model':<8}  {'FS':>6}  {'KLR':>6}  {'ARR':>6}  {'Utility':>7}  Visual (FS)")
p(SEP)

rows = [
    ("Baseline (no unlearning)",  "1.5B",  baseline_1b,  util(baseline_full)),
    ("NPO",                        "1.5B",  npo,           None),
    ("Gradient Ascent (GA)",       "1.5B",  ga,            None),
    ("GRPO only (Run 6)",          "1.5B",  grpo_run6,     None),
    ("SFT on ignorance phrases",   "1.5B",  sft_only,      None),
    ("SFT on SK facts",            "1.5B",  sft_sk,        None),
    ("SFT + GRPO (FINAL)",         "1.5B",  sft_grpo_1b,   0.70),
    ("8B SFT Stage 1 (v5)",        "8B",    sft_8b_v5,     None),
    ("8B SFT + GRPO v5 (FINAL)",   "8B",    sft_grpo_8b,   util(util_8b_v5)),
]

for name, model, d, u in rows:
    f_ = fs(d); k_ = klr(d); a_ = arr(d)
    u_ = f"{u*100:5.1f}%" if u is not None else "  N/A "
    vis = bar(f_) if f_ is not None else "  N/A    "
    f_s = fmt(f_) if f_ is not None else "  N/A "
    k_s = fmt(k_) if k_ is not None else "  N/A "
    a_s = fmt(a_) if a_ is not None else "  N/A "
    p(f"  {name:<30} {model:<8}  {f_s}  {k_s}  {a_s}  {u_:>7}  {vis}")

p(SEP)
p()

# ── TABLE 2: Per-level breakdown for key models ───────────────────────────────
p("TABLE 2: Per-Level Breakdown — Key Models")
p(SEP)
p(f"  {'Method':<30} {'Lvl':<4}  {'FS':>6}  {'KLR':>6}  {'ARR':>6}  Visual (FS)")
p(SEP)

key_models = [
    ("Baseline (1.5B)",       baseline_1b),
    ("SFT only (1.5B)",       sft_only),
    ("SFT+GRPO 1.5B FINAL",   sft_grpo_1b),
    ("8B SFT+GRPO v5 FINAL",  sft_grpo_8b),
]

for name, d in key_models:
    for lvl in ["L1", "L2", "L3"]:
        f_, k_, a_ = level(d, lvl)
        vis = bar(f_) if f_ is not None else "  N/A    "
        f_s = fmt(f_) if f_ is not None else "  N/A "
        k_s = fmt(k_) if k_ is not None else "  N/A "
        a_s = fmt(a_) if a_ is not None else "  N/A "
        label = name if lvl == "L1" else ""
        p(f"  {label:<30} {lvl:<4}  {f_s}  {k_s}  {a_s}  {vis}")
    p()
p(SEP)
p()

# ── TABLE 3: 8B iteration progress ────────────────────────────────────────────
p("TABLE 3: 8B Iteration Progress — SK Forget + Utility + OOD (Tom Clancy ARR)")
p(SEP)
p(f"  {'Version':<20} {'SK FS':>6}  {'SK KLR':>7}  {'SK ARR':>7}  {'Utility':>7}  {'TC-L1 ARR':>10}  {'TC-L2 ARR':>10}  {'TC-L3 ARR':>10}")
p(SEP)

iter_data = [
    ("8B SFT v5",        load("sft_8b_v5_merged_l123.json"),   util(load("utility_ood_8b_v3.json")),  tc_v3),
    ("8B SFT+GRPO v3",   load("sft_grpo_8b_v3_ckpt300_l123.json"), util(load("utility_ood_8b_v3.json")), tc_v3),
    ("8B SFT+GRPO v4",   load("sft_grpo_8b_v4_ckpt300_l123.json"), util(load("utility_ood_8b_v4.json")), tc_v4),
    ("8B SFT+GRPO v5",   sft_grpo_8b,                          util(util_8b_v5),                      tc_v5),
]

for name, d, u, tc in iter_data:
    f_ = fs(d); k_ = klr(d); a_ = arr(d)
    u_s = fmt(u) if u is not None else "  N/A "
    f_s = fmt(f_) if f_ is not None else "  N/A "
    k_s = fmt(k_) if k_ is not None else "  N/A "
    a_s = fmt(a_) if a_ is not None else "  N/A "
    tc1 = fmt(tc.get("L1",{}).get("answer_recall_rate")) if tc and "L1" in tc else "  N/A "
    tc2 = fmt(tc.get("L2",{}).get("answer_recall_rate")) if tc and "L2" in tc else "  N/A "
    tc3 = fmt(tc.get("L3",{}).get("answer_recall_rate")) if tc and "L3" in tc else "  N/A "
    p(f"  {name:<20} {f_s}  {k_s}   {a_s}   {u_s}  {tc1:>10}  {tc2:>10}  {tc3:>10}")

p(SEP)
p()

# ── TABLE 4: OOD specificity (Tom Clancy) across versions ─────────────────────
p("TABLE 4: OOD Specificity — Tom Clancy ARR by Level (should be HIGH = model still knows TC)")
p("  Note: For OOD subjects, ARR = 1.0 means model retains that knowledge (GOOD).")
p("        FS < 0.5 is acceptable for OOD because we WANT ARR to stay high.")
p(SEP)
p(f"  {'Version':<22} {'L1 ARR':>7}  {'L1 FS':>6}  {'L2 ARR':>7}  {'L2 FS':>6}  {'L3 ARR':>7}  {'L3 FS':>6}")
p(SEP)

for label, tc in [("8B SFT+GRPO v3", tc_v3), ("8B SFT+GRPO v4", tc_v4), ("8B SFT+GRPO v5", tc_v5)]:
    def g(lvl, key):
        return fmt(tc.get(lvl, {}).get(key)) if tc and lvl in tc else "  N/A "
    p(f"  {label:<22} {g('L1','answer_recall_rate'):>7}  {g('L1','forget_score'):>6}  {g('L2','answer_recall_rate'):>7}  {g('L2','forget_score'):>6}  {g('L3','answer_recall_rate'):>7}  {g('L3','forget_score'):>6}")

p(SEP)
p()

# ── SUMMARY BOX ───────────────────────────────────────────────────────────────
p(SEP2)
p("  KEY FINDINGS")
p(SEP2)
p("  1. GRPO alone (Run 6): FS=0.500 — variance collapse on deeply memorized facts")
p("  2. GA / NPO:            FS≈0.39-0.41 — minimal improvement over baseline (FS=0.33)")
p("  3. SFT on ignorance phrases: FS=1.000 (1.5B) — 100% forget, but utility drops to 70%")
p("  4. SFT + GRPO (1.5B):  FS=1.000, Utility=70%, OOD preserved — best 1.5B result")
p("  5. 8B SFT + GRPO v5:   FS=0.979, Utility=73%, TC-L1 ARR=90% — best overall result")
p("     • L1=0.9375, L2=1.000, L3=1.000 | KLR=0.000 across all levels")
p("     • Residual: 'author' token at L1 — generic knowledge, structurally irreducible")
p("  6. Retain format coverage principle: OOD specificity controlled by retain set format")
p(SEP2)

if SAVE_MD:
    out = RESULTS_DIR / "summary_table.md"
    with open(out, "w") as f:
        f.write("```\n" + "\n".join(lines) + "\n```\n")
    print(f"\nSaved to {out}")
