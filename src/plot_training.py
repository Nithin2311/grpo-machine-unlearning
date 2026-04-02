"""
Training metrics visualizer for GRPO unlearning runs.

Parses train_run*.log files and generates comprehensive plots.
Can be run at any time during or after training.

Usage:
    python src/plot_training.py                        # uses train_run2.log by default
    python src/plot_training.py --log train_run.log    # specify log file
    python src/plot_training.py --watch                # re-plot every 2 min (live monitor)
"""

import re
import json
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

PROJECT_DIR = Path(__file__).parent.parent
PLOTS_DIR   = PROJECT_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Log parser ────────────────────────────────────────────────────────────────

def parse_log(log_path: Path) -> pd.DataFrame:
    """Extract all step metrics from a GRPO training log into a DataFrame.

    trl 0.29.x omits the 'step' key from log dicts, so we infer step
    from the sequential order of dicts found in the log.
    """
    records = []
    # Match any dict that contains 'reward' — these are the step-level metrics
    pattern = re.compile(r"\{'loss':.*?'epoch':\s*[\d.]+\}", re.DOTALL)

    text = log_path.read_text(errors="ignore")
    for i, match in enumerate(pattern.finditer(text), start=1):
        try:
            raw = match.group().replace("'", '"').replace("True", "true").replace("False", "false")
            record = json.loads(raw)
            record["step"] = record.get("step", i)  # use embedded step or infer
            records.append(record)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).reset_index(drop=True)
    df["reward_smooth"] = df["reward"].rolling(window=10, min_periods=1).mean()
    return df


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_reward_overview(df: pd.DataFrame, ax: plt.Axes):
    """Total reward over training steps with smoothed trend."""
    ax.plot(df["step"], df["reward"], alpha=0.35, color="#5b9bd5", linewidth=1, label="Raw reward")
    ax.plot(df["step"], df["reward_smooth"], color="#1f4e79", linewidth=2, label="Smoothed (10-step)")
    ax.axhline(0, color="green", linestyle="--", linewidth=1, alpha=0.6, label="Target (0)")
    ax.fill_between(df["step"], df["reward"], 0,
                    where=(df["reward"] < 0), alpha=0.08, color="red")
    ax.set_title("Total GRPO Reward over Training", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.set_xlim(left=1)


def plot_reward_components(df: pd.DataFrame, ax: plt.Axes):
    """Three reward function components stacked on one plot."""
    cols = {
        "rewards/entity_leak_penalty_reward/mean": ("Entity Leak Penalty", "#e74c3c"),
        "rewards/plausible_ignorance_reward/mean": ("Plausible Ignorance",  "#f39c12"),
        "rewards/format_adherence_reward/mean":    ("Format Adherence",     "#27ae60"),
    }
    for col, (label, color) in cols.items():
        if col in df.columns:
            smooth = df[col].rolling(10, min_periods=1).mean()
            ax.plot(df["step"], smooth, label=label, color=color, linewidth=2)
            ax.plot(df["step"], df[col], alpha=0.2, color=color, linewidth=1)

    ax.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_title("Reward Function Breakdown", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=8)
    ax.set_xlim(left=1)


def plot_entity_leak_rate(df: pd.DataFrame, ax: plt.Axes):
    """Entity leak penalty over time — key unlearning signal."""
    col = "rewards/entity_leak_penalty_reward/mean"
    if col not in df.columns:
        return

    # Convert penalty to leak rate: penalty=-2.0 → all leaking, -0.5*n → none leaking
    # penalty = leaked * (-2.0) + not_leaked * (0.5)  per generation
    # Approximate leak rate from mean penalty
    smooth = df[col].rolling(10, min_periods=1).mean()
    ax.fill_between(df["step"], df[col], alpha=0.2, color="#e74c3c")
    ax.plot(df["step"], smooth, color="#c0392b", linewidth=2.5, label="Smoothed")
    ax.plot(df["step"], df[col], alpha=0.3, color="#e74c3c", linewidth=1)
    ax.axhline(-2.0, color="red",   linestyle=":", linewidth=1, label="-2.0 (all leaking)")
    ax.axhline( 0.5, color="green", linestyle=":", linewidth=1, label="+0.5 (none leaking)")
    ax.set_title("Entity Leak Penalty (Key Unlearning Signal)", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Leak Penalty")
    ax.legend(fontsize=8)
    ax.set_xlim(left=1)


def plot_entropy_and_kl(df: pd.DataFrame, ax: plt.Axes):
    """Entropy and KL divergence — model diversity and policy drift."""
    ax2 = ax.twinx()
    if "entropy" in df.columns:
        smooth_e = df["entropy"].rolling(10, min_periods=1).mean()
        ax.plot(df["step"], smooth_e, color="#8e44ad", linewidth=2, label="Entropy")
        ax.set_ylabel("Entropy", color="#8e44ad")
        ax.tick_params(axis="y", labelcolor="#8e44ad")

    if "kl" in df.columns:
        smooth_k = df["kl"].rolling(10, min_periods=1).mean()
        ax2.plot(df["step"], smooth_k, color="#16a085", linewidth=2,
                 linestyle="--", label="KL divergence")
        ax2.set_ylabel("KL Divergence", color="#16a085")
        ax2.tick_params(axis="y", labelcolor="#16a085")

    ax.set_title("Policy Entropy & KL Divergence", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_xlim(left=1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)


def plot_grad_norm(df: pd.DataFrame, ax: plt.Axes):
    """Gradient norm — training stability indicator."""
    if "grad_norm" not in df.columns:
        return
    smooth = df["grad_norm"].rolling(10, min_periods=1).mean()
    ax.plot(df["step"], df["grad_norm"], alpha=0.25, color="#e67e22", linewidth=1)
    ax.plot(df["step"], smooth, color="#d35400", linewidth=2, label="Smoothed")
    ax.set_title("Gradient Norm (Training Stability)", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Grad Norm")
    ax.legend(fontsize=8)
    ax.set_xlim(left=1)


def plot_completion_length(df: pd.DataFrame, ax: plt.Axes):
    """Mean completion length and clipped ratio over time."""
    if "completions/mean_length" not in df.columns:
        return
    ax2 = ax.twinx()
    smooth_len = df["completions/mean_length"].rolling(10, min_periods=1).mean()
    ax.plot(df["step"], smooth_len, color="#2980b9", linewidth=2, label="Mean length")
    ax.set_ylabel("Tokens", color="#2980b9")
    ax.tick_params(axis="y", labelcolor="#2980b9")

    if "completions/clipped_ratio" in df.columns:
        smooth_clip = df["completions/clipped_ratio"].rolling(10, min_periods=1).mean()
        ax2.plot(df["step"], smooth_clip, color="#e74c3c", linewidth=2,
                 linestyle="--", label="Clipped ratio")
        ax2.set_ylabel("Clipped Ratio", color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.set_ylim(0, 1)

    ax.set_title("Completion Length & Clipping", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_xlim(left=1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)


def plot_reward_std(df: pd.DataFrame, ax: plt.Axes):
    """Reward std and frac_reward_zero_std — exploration health."""
    if "reward_std" not in df.columns:
        return
    smooth = df["reward_std"].rolling(10, min_periods=1).mean()
    ax.fill_between(df["step"], smooth, alpha=0.2, color="#27ae60")
    ax.plot(df["step"], smooth, color="#1e8449", linewidth=2, label="Reward std (smoothed)")

    if "frac_reward_zero_std" in df.columns:
        ax2 = ax.twinx()
        smooth_z = df["frac_reward_zero_std"].rolling(10, min_periods=1).mean()
        ax2.plot(df["step"], smooth_z, color="#e74c3c", linewidth=1.5,
                 linestyle="--", label="Zero-std fraction")
        ax2.set_ylabel("Zero-std Fraction", color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.set_ylim(0, 1)
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(fontsize=8)
        ax2.legend(lines2, labels2, fontsize=8, loc="upper right")

    ax.set_title("Reward Diversity (Exploration Health)", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward Std")
    ax.set_xlim(left=1)


def plot_step_time(df: pd.DataFrame, ax: plt.Axes):
    """Step time over training."""
    if "step_time" not in df.columns:
        return
    smooth = df["step_time"].rolling(10, min_periods=1).mean()
    ax.bar(df["step"], df["step_time"], alpha=0.3, color="#7f8c8d", width=1)
    ax.plot(df["step"], smooth, color="#2c3e50", linewidth=2, label="Smoothed")
    ax.set_title("Step Time (seconds)", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=8)
    ax.set_xlim(left=1)


def plot_reward_bar_snapshot(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: average reward per component across three training phases."""
    thirds = len(df) // 3
    if thirds == 0:
        thirds = max(1, len(df))

    phases = {
        "Early\n(steps 1–{})".format(thirds):           df.iloc[:thirds],
        "Mid\n(steps {}–{})".format(thirds, 2*thirds):  df.iloc[thirds:2*thirds],
        "Late\n(steps {}+)".format(2*thirds):           df.iloc[2*thirds:],
    }

    components = {
        "Entity Leak":  "rewards/entity_leak_penalty_reward/mean",
        "Ignorance":    "rewards/plausible_ignorance_reward/mean",
        "Format":       "rewards/format_adherence_reward/mean",
        "Total":        "reward",
    }

    x = range(len(phases))
    width = 0.18
    colors = ["#e74c3c", "#f39c12", "#27ae60", "#2980b9"]

    for i, (comp_label, col) in enumerate(components.items()):
        if col not in df.columns:
            continue
        vals = [phase_df[col].mean() for phase_df in phases.values()]
        offset = (i - 1.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label=comp_label, color=colors[i], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(list(phases.keys()), fontsize=8)
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)
    ax.set_title("Reward Components by Training Phase", fontweight="bold")
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=7, ncol=2)


# ── Main dashboard ────────────────────────────────────────────────────────────

def generate_dashboard(log_path: Path, out_prefix: str = "training_dashboard"):
    df = parse_log(log_path)
    if df.empty:
        print(f"No step data found yet in {log_path}")
        return None

    n_steps = df["step"].max()
    print(f"  Parsed {len(df)} steps (max step: {n_steps})")

    # ── 3x3 dashboard ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"GRPO Unlearning — Qwen2.5-1.5B | Stephen King | {n_steps}/300 steps",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    plot_reward_overview(df,       fig.add_subplot(gs[0, :2]))   # wide top-left
    plot_reward_bar_snapshot(df,   fig.add_subplot(gs[0, 2]))    # top-right bar
    plot_reward_components(df,     fig.add_subplot(gs[1, :2]))   # wide middle-left
    plot_entity_leak_rate(df,      fig.add_subplot(gs[1, 2]))    # middle-right
    plot_entropy_and_kl(df,        fig.add_subplot(gs[2, 0]))    # bottom-left
    plot_grad_norm(df,             fig.add_subplot(gs[2, 1]))    # bottom-mid
    plot_completion_length(df,     fig.add_subplot(gs[2, 2]))    # bottom-right

    out = PLOTS_DIR / f"{out_prefix}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard saved: {out}")

    # ── Separate detailed charts ───────────────────────────────────────────────
    for name, fn in [
        ("reward_over_time",    plot_reward_overview),
        ("reward_components",   plot_reward_components),
        ("entity_leak",         plot_entity_leak_rate),
        ("entropy_kl",          plot_entropy_and_kl),
        ("grad_norm",           plot_grad_norm),
        ("completion_length",   plot_completion_length),
        ("reward_diversity",    plot_reward_std),
        ("step_time",           plot_step_time),
        ("phase_bar_chart",     plot_reward_bar_snapshot),
    ]:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        fn(df, ax2)
        fig2.tight_layout()
        p = PLOTS_DIR / f"{name}.png"
        fig2.savefig(p, bbox_inches="tight")
        plt.close(fig2)

    print(f"  {len(list(PLOTS_DIR.glob('*.png')))} total plots in {PLOTS_DIR}/")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",   default="../train_run2.log", help="Path to training log")
    parser.add_argument("--watch", action="store_true",         help="Re-plot every 2 minutes")
    parser.add_argument("--interval", type=int, default=120,    help="Watch interval in seconds")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = Path(__file__).parent / log_path

    if args.watch:
        print(f"Live monitoring {log_path} — updating every {args.interval}s. Ctrl+C to stop.")
        while True:
            print(f"\n[{pd.Timestamp.now().strftime('%H:%M:%S')}] Generating plots...")
            generate_dashboard(log_path)
            time.sleep(args.interval)
    else:
        generate_dashboard(log_path)
