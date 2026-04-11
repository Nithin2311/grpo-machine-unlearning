"""
Comprehensive results visualization — all 1.5B and 8B experimental runs.

Generates 5 publication-ready figures saved to results/plots/:
  1. methods_comparison.png     — All methods FS/KLR/ARR bar chart (1.5B)
  2. level_breakdown.png        — L1/L2/L3 breakdown for best models
  3. 8b_iteration_progress.png  — 8B v1→v5 improvement story
  4. ood_progression.png        — OOD Tom Clancy ARR across 8B versions
  5. utility_comparison.png     — Utility preservation across all methods

Usage:
  python3.11 src/plot_all_results.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────────────
C_FS   = "#2196F3"   # blue  — Forget Score
C_KLR  = "#F44336"   # red   — Keyword Leak Rate
C_ARR  = "#FF9800"   # orange — Answer Recall Rate
C_UTIL = "#4CAF50"   # green — Utility

GREY   = "#9E9E9E"
DARK   = "#212121"

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — All 1.5B Methods Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_methods_comparison():
    methods = [
        ("Baseline\n(1.5B)",      0.332,  0.843, 0.493, None),
        ("NPO\n(1.5B)",           0.389,  1.000, 0.222, None),
        ("GRPO Alone\n(1.5B)",    0.500,  0.500, 0.500, None),
        ("GA + Retain\n(1.5B)",   0.408,  0.912, 0.271, None),
        ("SFT Alone\n(1.5B)",     1.000,  0.000, 0.000, "1.5B best"),
        ("SFT+GRPO\n(1.5B)",      1.000,  0.000, 0.000, "1.5B best"),
        ("8B Baseline\n(Llama-3.1-8B)", 0.2569, 0.6394, 0.8468, None),
        ("SFT+GRPO\n(8B v5)",     0.979,  0.000, 0.042, "8B best"),
    ]

    labels = [m[0] for m in methods]
    fs     = [m[1] for m in methods]
    klr    = [m[2] for m in methods]
    arr    = [m[3] for m in methods]
    tags   = [m[4] for m in methods]

    x = np.arange(len(labels))
    w = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    b1 = ax.bar(x - w, fs,  w, label="Forget Score (↑ better)", color=C_FS,   alpha=0.88, zorder=3)
    b2 = ax.bar(x,     klr, w, label="Keyword Leak Rate (↓ better)", color=C_KLR, alpha=0.88, zorder=3)
    b3 = ax.bar(x + w, arr, w, label="Answer Recall Rate (↓ better)", color=C_ARR, alpha=0.88, zorder=3)

    # Highlight best models
    for i, tag in enumerate(tags):
        if tag:
            for bar in [b1[i], b2[i], b3[i]]:
                bar.set_edgecolor(DARK)
                bar.set_linewidth(2.0)

    # Target line
    ax.axhline(1.0, color=C_FS, linestyle="--", linewidth=1, alpha=0.4, zorder=2)

    # Divider between 1.5B and 8B
    ax.axvline(6.5, color=GREY, linestyle=":", linewidth=1.5, zorder=2)
    ax.text(6.6, 1.02, "8B Model →", fontsize=8, color=GREY, va="bottom")
    ax.text(6.4, 1.02, "← 1.5B Model", fontsize=8, color=GREY, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("All Methods: Forget Score, Keyword Leak Rate, Answer Recall Rate\n"
                 "Stephen King unlearning — RWKU benchmark (L1+L2+L3 combined)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    # RMU note
    ax.annotate("RMU methods excluded:\nFS=1.0 false positive\n(incoherent output)",
                xy=(0.5, 0.72), xycoords="axes fraction",
                fontsize=7.5, color=GREY,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREY, alpha=0.8))

    save(fig, "methods_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — L1 / L2 / L3 Level Breakdown for Best Models
# ══════════════════════════════════════════════════════════════════════════════

def plot_level_breakdown():
    data = {
        "Baseline (1.5B)": {
            "L1": (0.3125, 0.875,  0.500),
            "L2": (0.3500, 0.900,  0.300),
            "L3": (0.3318, 0.813,  0.517),
        },
        "Baseline (8B)": {
            "L1": (0.5000, 0.125,  0.875),
            "L2": (0.1500, 1.000,  0.700),
            "L3": (0.1207, 0.793,  0.966),
        },
        "GRPO Alone (1.5B)": {
            "L1": (0.500,  0.500,  0.500),
            "L2": (0.200,  1.000,  0.600),
            "L3": (0.483,  0.517,  0.517),
        },
        "SFT+GRPO (1.5B)\n— Best": {
            "L1": (1.000,  0.000,  0.000),
            "L2": (1.000,  0.000,  0.000),
            "L3": (1.000,  0.000,  0.000),
        },
        "SFT+GRPO (8B v5)\n— Best": {
            "L1": (0.9375, 0.000,  0.125),
            "L2": (1.000,  0.000,  0.000),
            "L3": (1.000,  0.000,  0.000),
        },
    }

    levels  = ["L1\n(Direct)", "L2\n(Paraphrased)", "L3\n(Adversarial)"]
    models  = list(data.keys())
    n_mod   = len(models)
    n_lvl   = len(levels)
    x       = np.arange(n_lvl)
    w       = 0.18
    offsets = np.linspace(-(n_mod-1)*w/2, (n_mod-1)*w/2, n_mod)

    colors  = ["#90CAF9", "#FFB74D", "#EF9A9A", "#A5D6A7", "#2196F3"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    fig.patch.set_facecolor("white")
    metrics     = ["Forget Score (↑)", "Keyword Leak Rate (↓)", "Answer Recall Rate (↓)"]
    metric_idx  = [0, 1, 2]
    metric_cols = [C_FS, C_KLR, C_ARR]

    for col, (ax, metric, midx) in enumerate(zip(axes, metrics, metric_idx)):
        for i, (model, lvl_data) in enumerate(data.items()):
            vals = [lvl_data[l.split("\n")[0]][midx] for l in levels]
            bars = ax.bar(x + offsets[i], vals, w,
                          label=model.replace("\n", " "), color=colors[i], alpha=0.88,
                          edgecolor=DARK if "Best" in model else "none", linewidth=1.5,
                          zorder=3)

        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("Score", fontsize=10)

    handles = [mpatches.Patch(color=colors[i], label=m.replace("\n— Best","").replace("\n"," "))
               for i, m in enumerate(models)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
    fig.suptitle("Level-by-Level Breakdown: L1 (Direct) → L2 (Paraphrased) → L3 (Adversarial)\n"
                 "Stephen King unlearning — bold outline = best model per scale",
                 fontsize=12, fontweight="bold", y=1.02)

    save(fig, "level_breakdown.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — 8B Iteration Progress (v3 → v4 → v5)
# ══════════════════════════════════════════════════════════════════════════════

def plot_8b_iterations():
    versions = ["v3\n(ALPHA=0.35\n200 steps)",
                "v4\n(ALPHA=0.45\n200 steps)",
                "v5 ★\n(ALPHA=0.45\n300 steps)"]
    fs_vals   = [0.9375, 0.8984, 0.9792]
    klr_vals  = [0.000,  0.000,  0.000 ]
    arr_vals  = [0.125,  0.203,  0.042 ]
    util_vals = [0.68,   0.70,   0.73  ]
    ood_arr   = [0.308,  0.714,  0.680 ]  # combined TC ARR

    x = np.arange(len(versions))
    w = 0.18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    # Left: SK unlearning metrics
    ax1.bar(x - w, fs_vals,  w, label="Forget Score (↑)", color=C_FS,  alpha=0.88, zorder=3)
    ax1.bar(x,     klr_vals, w, label="KLR (↓)",          color=C_KLR, alpha=0.88, zorder=3)
    ax1.bar(x + w, arr_vals, w, label="ARR (↓)",          color=C_ARR, alpha=0.88, zorder=3)
    ax1.axhline(1.0, color=C_FS, linestyle="--", linewidth=1, alpha=0.4)
    ax1.set_xticks(x); ax1.set_xticklabels(versions, fontsize=9)
    ax1.set_ylim(0, 1.12)
    ax1.set_title("SK Unlearning Scores\nby 8B Training Version", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=10)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, zorder=1); ax1.set_axisbelow(True)

    # Right: Utility + OOD retention
    ax2.plot(versions, util_vals, "o-", color=C_UTIL, linewidth=2.5, markersize=9,
             label="Utility (↑ better)", zorder=4)
    ax2.plot(versions, ood_arr,   "s--", color="#9C27B0", linewidth=2.5, markersize=9,
             label="OOD ARR — Tom Clancy (↑ = model still knows TC)", zorder=4)
    ax2.axhline(0.70, color=C_UTIL, linestyle=":", linewidth=1, alpha=0.5)
    ax2.text(2.05, 0.70, "70% target", fontsize=8, color=C_UTIL, va="center")
    for xi, (u, o) in enumerate(zip(util_vals, ood_arr)):
        ax2.annotate(f"{u:.0%}", (xi, u), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, color=C_UTIL, fontweight="bold")
        ax2.annotate(f"{o:.0%}", (xi, o), textcoords="offset points", xytext=(0, -16),
                     ha="center", fontsize=9, color="#9C27B0")
    ax2.set_xticks(range(len(versions))); ax2.set_xticklabels(versions, fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Utility Preservation & OOD Retention\nby 8B Training Version", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=10)
    ax2.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax2.grid(alpha=0.3, zorder=1); ax2.set_axisbelow(True)

    fig.suptitle("8B Model Training Iterations — Balancing Unlearning, Utility, and OOD Specificity",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "8b_iteration_progress.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — OOD Tom Clancy ARR by Level (v3 vs v4 vs v5)
# ══════════════════════════════════════════════════════════════════════════════

def plot_ood_progression():
    # ARR = model correctly answers TC questions (higher = better OOD retention)
    ood_data = {
        "v3\n(L1 OOD retain only)":      {"L1": 0.900, "L2": 0.000, "L3": 0.000},
        "v4\n(L1+L2 OOD retain)":        {"L1": 0.950, "L2": 0.615, "L3": 0.575},
        "v5 ★\n(L1+L2 OOD retain\n300 steps)": {"L1": 0.900, "L2": 0.539, "L3": 0.600},
    }
    levels  = ["L1\n(Direct)", "L2\n(Paraphrased)", "L3\n(Adversarial)"]
    colors  = ["#90CAF9", "#42A5F5", "#1565C0"]
    x       = np.arange(len(levels))
    w       = 0.24
    offsets = [-w, 0, w]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")

    for i, (version, lvl_data) in enumerate(ood_data.items()):
        vals = [lvl_data[l.split("\n")[0]] for l in levels]
        bars = ax.bar(x + offsets[i], vals, w, label=version.replace("\n"," "),
                      color=colors[i], alpha=0.88, zorder=3,
                      edgecolor=DARK if "★" in version else "none", linewidth=1.5)
        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(0.9, color=GREY, linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2.42, 0.91, "v3 L1 baseline", fontsize=7.5, color=GREY)

    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Tom Clancy Answer Recall Rate\n(higher = model still knows Tom Clancy)", fontsize=10)
    ax.set_title("OOD Knowledge Retention — Tom Clancy\n"
                 "Adding L2 retain examples (v4) restored paraphrased & adversarial OOD knowledge",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=1); ax.set_axisbelow(True)

    ax.annotate("v3: only L1 OOD\nretain examples\n→ L2/L3 refused",
                xy=(1.0, 0.02), xycoords="data",
                xytext=(0.35, 0.25), textcoords="axes fraction",
                fontsize=8, color="#F44336",
                arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.2))

    save(fig, "ood_progression.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Utility Preservation Across All Methods
# ══════════════════════════════════════════════════════════════════════════════

def plot_utility_comparison():
    methods  = [
        "Baseline\n(1.5B)",
        "GRPO Alone\n(1.5B)",
        "GA+Retain\n(1.5B)",
        "SFT+GRPO\n(1.5B)",
        "8B v1\n(over-refusal)",
        "8B v2",
        "8B v3",
        "8B v4",
        "8B v5 ★",
    ]
    utility  = [0.75, 0.73, None, 0.70, 0.00, 0.68, 0.68, 0.70, 0.73]
    fs_vals  = [0.332, 0.500, 0.408, 1.000, 1.000, 1.000, 0.938, 0.898, 0.979]
    colors   = (["#90CAF9"]*4 + ["#FFCDD2"] + ["#81C784"]*3 + ["#2E7D32"])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(methods))
    util_plot = [u if u is not None else 0 for u in utility]
    bars = ax.bar(x, util_plot, 0.55, color=colors, alpha=0.88, zorder=3,
                  edgecolor=[DARK if "★" in m else "none" for m in methods], linewidth=1.5)

    # FS overlay as line
    ax2 = ax.twinx()
    ax2.plot(x, fs_vals, "D--", color=C_FS, linewidth=2, markersize=8,
             label="Forget Score (right axis)", zorder=5, alpha=0.9)
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel("Forget Score", fontsize=10, color=C_FS)
    ax2.tick_params(axis="y", colors=C_FS)

    # Annotations
    for xi, (u, m) in enumerate(zip(utility, methods)):
        if u is None:
            ax.text(xi, 0.03, "N/A", ha="center", va="bottom", fontsize=8, color=GREY)
        elif u == 0:
            ax.text(xi, 0.03, "0%\n(over-refusal)", ha="center", va="bottom",
                    fontsize=7.5, color="#C62828")
        else:
            ax.text(xi, u + 0.01, f"{u:.0%}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.axhline(0.70, color=C_UTIL, linestyle="--", linewidth=1.5, alpha=0.6, zorder=2)
    ax.text(-0.45, 0.715, "70% target", fontsize=8.5, color=C_UTIL)
    ax.axhline(0.40, color=C_KLR, linestyle=":", linewidth=1, alpha=0.5, zorder=2)
    ax.text(-0.45, 0.415, "over-refusal threshold", fontsize=7.5, color=C_KLR)

    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Utility Accuracy (RWKU utility_general)", fontsize=10)
    ax.set_title("Utility Preservation vs Forget Score — All Methods\n"
                 "Blue bars = 1.5B | Green bars = 8B | Red = failure (over-refusal)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=1); ax.set_axisbelow(True)

    fs_patch = mpatches.Patch(color=C_FS, label="Forget Score (line, right axis)")
    ax2.legend(handles=[fs_patch], loc="upper left", fontsize=9, framealpha=0.9)

    save(fig, "utility_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Run all plots
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating all plots...")
    plot_methods_comparison()
    plot_level_breakdown()
    plot_8b_iterations()
    plot_ood_progression()
    plot_utility_comparison()
    print(f"\nAll plots saved to: {OUT_DIR}/")
    print("  1. methods_comparison.png")
    print("  2. level_breakdown.png")
    print("  3. 8b_iteration_progress.png")
    print("  4. ood_progression.png")
    print("  5. utility_comparison.png")
