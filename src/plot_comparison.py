"""
plot_comparison.py — Generate updated comparison plots for all 1B model runs.

Produces:
  results/plots/all_runs_comparison.png  — bar chart of FS/KLR/ARR for all methods
  results/plots/method_tradeoff.png      — scatter: FS vs Utility Score (if available)
  results/plots/npo_curve.png            — NPO checkpoint progression

Usage:
  python3 src/plot_comparison.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS = "results"
PLOTS   = "results/plots"
os.makedirs(PLOTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Load all scores
# ---------------------------------------------------------------------------

def load(path, subkey=None):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if subkey:
        d = d.get(subkey, {})
    return d

gda      = load(f"{RESULTS}/gda_eval.json", subkey="GDA-final")
baseline_full = load(f"{RESULTS}/full_eval_baseline.json")
winner_full   = load(f"{RESULTS}/full_eval_run6_ckpt300.json")

# Fallback baseline (from known values if full eval not done yet)
baseline_fs  = baseline_full["forget_score"]  if baseline_full else 0.3889
baseline_klr = baseline_full["keyword_leak_rate"] if baseline_full else 0.8333
baseline_arr = baseline_full["answer_recall_rate"] if baseline_full else 0.3889
baseline_us  = baseline_full.get("utility_score") if baseline_full else None

# All methods for the main comparison bar chart
# Format: (label, fs, klr, arr, group)
entries = [
    ("Baseline",       baseline_fs,  baseline_klr, baseline_arr, "baseline"),
    ("GDA\nfinal",     gda["forget_score"] if gda else 0.4444,
                       gda["keyword_leak_rate"] if gda else 0.7778,
                       gda["answer_recall_rate"] if gda else 0.3333, "gda"),
    ("Run5\nckpt100",  0.2778, 0.9444, 0.5000, "run5"),
    ("Run6\nckpt100",  0.3333, 0.9444, 0.3889, "run6"),
    ("Run6\nckpt200",  0.3333, 0.9444, 0.3889, "run6"),
    ("Run6\nckpt300",  0.2778, 0.9444, 0.5000, "run6"),
    ("NPO\nckpt100",   0.3333, 0.9444, 0.3889, "npo"),
    ("NPO\nckpt200",   0.3611, 1.0000, 0.2778, "npo"),
    ("NPO\nckpt300",   0.3889, 0.9444, 0.2778, "npo"),
    ("NPO\nckpt400",   0.4167, 1.0000, 0.1667, "npo"),
    ("NPO\nfinal",     0.3889, 1.0000, 0.2222, "npo"),
]

# Override run6/ckpt300 with full eval if available
if winner_full:
    fd = winner_full.get("forget_detail", winner_full)
    for i, e in enumerate(entries):
        if e[0] == "Run6\nckpt300":
            entries[i] = ("Run6\nckpt300",
                          winner_full["forget_score"],
                          fd.get("keyword_leak_rate", 0.9444),
                          fd.get("answer_recall_rate", 0.5000),
                          "run6")

GROUP_COLORS = {
    "baseline": "#6c757d",
    "gda":      "#fd7e14",
    "run5":     "#6f42c1",
    "run6":     "#0d6efd",
    "npo":      "#198754",
}

WINNER_IDX = [i for i, e in enumerate(entries) if e[0] == "Run6\nckpt300"][0]

# ---------------------------------------------------------------------------
# Plot 1: All-runs comparison bar chart (FS / KLR / ARR)
# ---------------------------------------------------------------------------

labels = [e[0] for e in entries]
fs_vals  = [e[1] for e in entries]
klr_vals = [e[2] for e in entries]
arr_vals = [e[3] for e in entries]
colors   = [GROUP_COLORS[e[4]] for e in entries]

x = np.arange(len(labels))
w = 0.26

fig, ax = plt.subplots(figsize=(15, 5.5))
bars_fs  = ax.bar(x - w, fs_vals,  w, label="Forget Score (↑ better)",      color=colors, alpha=0.95)
bars_klr = ax.bar(x,     klr_vals, w, label="Keyword Leak Rate (↓ better)",  color=colors, alpha=0.55, hatch="//")
bars_arr = ax.bar(x + w, arr_vals, w, label="Answer Recall Rate (↓ better)", color=colors, alpha=0.35, hatch="xx")

# Highlight winner
for offset, bars in [(-w, bars_fs), (0, bars_klr), (w, bars_arr)]:
    bars[WINNER_IDX].set_edgecolor("gold")
    bars[WINNER_IDX].set_linewidth(2.5)

ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="0.5 reference")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.08)
ax.set_title("All Methods — Forget Score / KLR / ARR (Qwen2.5-1.5B-Instruct, Stephen King)", fontsize=11)

# Legend: metric style + group colour
metric_patches = [
    mpatches.Patch(facecolor="gray", alpha=0.95, label="Forget Score (FS) ↑"),
    mpatches.Patch(facecolor="gray", alpha=0.55, hatch="//", label="Keyword Leak Rate (KLR) ↓"),
    mpatches.Patch(facecolor="gray", alpha=0.35, hatch="xx", label="Answer Recall Rate (ARR) ↓"),
]
group_patches = [mpatches.Patch(color=c, label=g.upper()) for g, c in GROUP_COLORS.items()]
ax.legend(handles=metric_patches + group_patches, fontsize=7.5, ncol=4, loc="upper right")

# Annotate FS values on top
for i, bar in enumerate(bars_fs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{fs_vals[i]:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

# Star winner
ax.text(x[WINNER_IDX] - w, fs_vals[WINNER_IDX] + 0.055, "★ WINNER",
        ha="center", fontsize=7, color="gold", fontweight="bold")

plt.tight_layout()
out = f"{PLOTS}/all_runs_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 2: NPO checkpoint progression curve
# ---------------------------------------------------------------------------

npo_steps = [100, 200, 300, 400, 500]
npo_fs    = [0.3333, 0.3611, 0.3889, 0.4167, 0.3889]
npo_klr   = [0.9444, 1.0000, 0.9444, 1.0000, 1.0000]
npo_arr   = [0.3889, 0.2778, 0.2778, 0.1667, 0.2222]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(npo_steps, npo_fs,  "o-",  color="#198754", label="Forget Score (FS) ↑", linewidth=2)
ax.plot(npo_steps, npo_klr, "s--", color="#dc3545", label="Keyword Leak Rate (KLR) ↓", linewidth=1.5)
ax.plot(npo_steps, npo_arr, "^--", color="#0dcaf0", label="Answer Recall Rate (ARR) ↓", linewidth=1.5)

# Baseline reference lines
ax.axhline(baseline_fs,  color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"Baseline FS={baseline_fs:.3f}")
ax.axhline(0.2778, color="#0d6efd", linestyle=":", linewidth=1, alpha=0.7, label="Run6 ckpt300 FS=0.278 (winner)")

ax.set_xlabel("NPO Training Steps")
ax.set_ylabel("Score")
ax.set_title("NPO Checkpoint Progression", fontsize=11)
ax.set_xticks(npo_steps)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = f"{PLOTS}/npo_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 3: FS vs Utility Score scatter (only if full evals exist)
# ---------------------------------------------------------------------------

scatter_data = []
if baseline_us is not None:
    scatter_data.append(("Baseline", baseline_fs, baseline_us, "baseline"))
if winner_full and winner_full.get("utility_score") is not None:
    scatter_data.append(("Run6 ckpt300\n(winner)", winner_full["forget_score"],
                         winner_full["utility_score"], "run6"))
# Add run3 from old eval
old = load(f"{RESULTS}/stephen_king_scores.json")
if old and old.get("utility_score"):
    scatter_data.append(("Run3 ckpt300\n(old eval)", old["forget_score"], old["utility_score"], "run5"))

if len(scatter_data) >= 2:
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, fs, us, grp in scatter_data:
        ax.scatter(fs, us, s=120, color=GROUP_COLORS[grp], zorder=3)
        ax.annotate(label, (fs, us), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xlabel("Forget Score (higher = better unlearning)")
    ax.set_ylabel("Utility Score (higher = better retention)")
    ax.set_title("Forget Score vs Utility Score\n(ideal: top-right)", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out = f"{PLOTS}/method_tradeoff.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
else:
    print("Skipping tradeoff scatter — full_eval results not ready yet.")

print("\nAll plots done. Check results/plots/")
