"""
Multi-run training comparison — overlays Run 1-4 entity_leak and reward signals.
"""
import re, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
PLOTS_DIR   = PROJECT_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def parse_log(log_path: Path):
    if not log_path.exists():
        return pd.DataFrame()
    pattern = re.compile(r"\{'loss':.*?'epoch':\s*[\d.]+\}", re.DOTALL)
    text = log_path.read_text(errors="ignore")
    records = []
    for i, m in enumerate(pattern.finditer(text), start=1):
        try:
            raw = m.group().replace("'",'"').replace("True","true").replace("False","false")
            r = json.loads(raw)
            r["step"] = r.get("step", i)
            records.append(r)
        except:
            pass
    return pd.DataFrame(records) if records else pd.DataFrame()

runs = {
    "Run 1": PROJECT_DIR / "train_run.log",
    "Run 2": PROJECT_DIR / "train_run2.log",
    "Run 3": PROJECT_DIR / "train_run3.log",
    "Run 4": PROJECT_DIR / "train_run4.log",
}
colors = {"Run 1":"#e74c3c","Run 2":"#f39c12","Run 3":"#3498db","Run 4":"#2ecc71"}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("GRPO Unlearning — All Runs Comparison\nQwen2.5-1.5B | Stephen King | RWKU",
             fontsize=13, fontweight="bold")

ax_reward, ax_leak, ax_std, ax_grad = axes.flat

for name, path in runs.items():
    df = parse_log(path)
    if df.empty:
        continue
    c = colors[name]
    label = f"{name} ({len(df)} steps)"
    s = df["step"]

    if "reward" in df:
        smooth = df["reward"].rolling(15, min_periods=1).mean()
        ax_reward.plot(s, smooth, color=c, linewidth=2, label=label)

    if "rewards/entity_leak_penalty_reward/mean" in df:
        smooth = df["rewards/entity_leak_penalty_reward/mean"].rolling(15, min_periods=1).mean()
        ax_leak.plot(s, smooth, color=c, linewidth=2, label=label)

    if "reward_std" in df:
        smooth = df["reward_std"].rolling(15, min_periods=1).mean()
        ax_std.plot(s, smooth, color=c, linewidth=2, label=label)

    if "grad_norm" in df:
        smooth = df["grad_norm"].rolling(15, min_periods=1).mean()
        ax_grad.plot(s, smooth, color=c, linewidth=2, label=label)

for ax, title, ylabel in [
    (ax_reward, "Total GRPO Reward (smoothed 15-step)", "Reward"),
    (ax_leak,   "Entity Leak Penalty (key unlearning signal)", "Mean Leak Reward"),
    (ax_std,    "Reward Std (exploration health)", "Reward Std"),
    (ax_grad,   "Gradient Norm (stability)", "Grad Norm"),
]:
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

ax_leak.axhline(-2.0, color="red",   linestyle=":", linewidth=1, alpha=0.7, label="-2.0 (all leak)")
ax_leak.axhline( 0.5, color="green", linestyle=":", linewidth=1, alpha=0.7, label="+0.5 (no leak)")
ax_reward.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
out = PLOTS_DIR / "all_runs_comparison.png"
fig.savefig(out, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {out}")
