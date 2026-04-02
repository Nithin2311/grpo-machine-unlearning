"""
AdaptiveGRPOCallback — dynamic hyperparameter adjustment every N steps.

Reads the last N steps of training metrics from trainer.state.log_history
and adjusts beta, learning rate, and grad norm clipping based on the
observed training health signals.

Adaptation rules (applied every `check_every` steps):
  1. MODE COLLAPSE    frac_zero_std > 0.4          → increase beta (force diversity)
  2. LOW VARIANCE     reward_std < 0.2             → increase beta (push exploration)
  3. STUCK POLICY     entity_leak == -2.0 for 10+  → decrease beta (loosen constraint)
  4. GOOD PROGRESS    reward trend improving        → hold beta, log confirmation
  5. GRAD SPIKE       avg_grad_norm > 1.5          → reduce learning rate by 20%
  6. STABLE + FAST    grad_norm < 0.3 & improving  → increase lr slightly (up to init)

Each adjustment is printed to console and written to an adjustment log,
making the adaptive trajectory fully reproducible for the report.
"""

from transformers import TrainerCallback
from pathlib import Path


# ── Bounds ────────────────────────────────────────────────────────────────────
BETA_MIN  = 0.005
BETA_MAX  = 0.30
LR_FLOOR  = 1e-7   # never drop lr below this


class AdaptiveGRPOCallback(TrainerCallback):
    """
    Dynamically adjusts GRPO hyperparameters every `check_every` steps.

    Usage:
        cb = AdaptiveGRPOCallback(check_every=10)
        trainer = GRPOTrainer(...)
        cb.register(trainer)          # back-reference so callback can mutate trainer
        trainer.add_callback(cb)
        trainer.train()
    """

    def __init__(self, check_every: int = 10, log_path: str = None,
                 init_lr: float = 5e-6):
        self.check_every  = check_every
        self.log_path     = Path(log_path) if log_path else None
        self.init_lr      = init_lr        # ceiling for LR recovery
        self.trainer      = None           # set via register()
        self.adjustments  = []             # full history for post-hoc analysis

    def register(self, trainer):
        """Call after trainer is created, before trainer.train()."""
        self.trainer = trainer

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_recent(self, key: str, n: int) -> list:
        history = self.trainer.state.log_history or []
        return [h[key] for h in history[-n:] if key in h]

    def _current_lr(self) -> float:
        try:
            return self.trainer.optimizer.param_groups[0]["lr"]
        except Exception:
            return self.init_lr

    def _set_lr(self, new_lr: float):
        new_lr = max(new_lr, LR_FLOOR)
        for pg in self.trainer.optimizer.param_groups:
            pg["lr"] = new_lr
        self.trainer.args.learning_rate = new_lr

    def _set_beta(self, new_beta: float):
        new_beta = max(BETA_MIN, min(BETA_MAX, new_beta))
        self.trainer.args.beta = new_beta
        # GRPOTrainer caches beta as self.beta — update both
        if hasattr(self.trainer, "beta"):
            self.trainer.beta = new_beta

    def _log(self, step: int, messages: list, metrics: dict):
        header = f"\n[Step {step:>4d}] ADAPTIVE ADJUSTMENT"
        body   = "\n".join(f"  • {m}" for m in messages)
        stats  = (f"  ↳ reward={metrics['avg_reward']:+.3f}  "
                  f"std={metrics['avg_std']:.3f}  "
                  f"zero_std={metrics['avg_zero']:.2f}  "
                  f"leak={metrics['avg_leak']:+.3f}  "
                  f"grad={metrics['avg_grad']:.3f}  "
                  f"beta={metrics['new_beta']:.4f}  "
                  f"lr={metrics['new_lr']:.2e}")
        full = f"{header}\n{body}\n{stats}"
        print(full)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(full + "\n")
        self.adjustments.append({"step": step, **metrics, "changes": messages})

    def _log_no_change(self, step: int, avg_reward: float, beta: float, lr: float):
        print(f"[Step {step:>4d}] OK — reward={avg_reward:+.3f}  "
              f"beta={beta:.4f}  lr={lr:.2e}  (no adjustment)")

    # ── Core adaptation logic ─────────────────────────────────────────────────

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.trainer is None:
            return
        step = state.global_step
        if step == 0 or step % self.check_every != 0:
            return

        n = self.check_every
        rewards     = self._get_recent("reward",                                    n)
        stds        = self._get_recent("reward_std",                                n)
        zero_stds   = self._get_recent("frac_reward_zero_std",                      n)
        grad_norms  = self._get_recent("grad_norm",                                 n)
        leaks       = self._get_recent("rewards/entity_leak_penalty_reward/mean",   n)

        if not rewards:
            return

        avg_reward = sum(rewards) / len(rewards)
        avg_std    = sum(stds)       / len(stds)       if stds       else 0.0
        avg_zero   = sum(zero_stds)  / len(zero_stds)  if zero_stds  else 0.0
        avg_grad   = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        avg_leak   = sum(leaks)      / len(leaks)       if leaks      else -2.0

        # Reward trend: positive = improving (reward going up toward 0)
        trend = (rewards[-1] - rewards[0]) / max(len(rewards), 1) if len(rewards) > 1 else 0.0

        cur_beta = self.trainer.beta if hasattr(self.trainer, "beta") else args.beta
        cur_lr   = self._current_lr()
        new_beta = cur_beta
        new_lr   = cur_lr
        changes  = []

        # ── Rule 1: Mode collapse → increase beta to force diverse updates ────
        if avg_zero > 0.4:
            new_beta = cur_beta * 1.25
            changes.append(
                f"Mode collapse (zero_std={avg_zero:.2f}) "
                f"→ beta {cur_beta:.4f} → {new_beta:.4f}"
            )

        # ── Rule 2: Low reward variance → gently raise beta ──────────────────
        elif avg_std < 0.20 and avg_zero > 0.15:
            new_beta = cur_beta * 1.12
            changes.append(
                f"Low reward variance (std={avg_std:.3f}) "
                f"→ beta {cur_beta:.4f} → {new_beta:.4f}"
            )

        # ── Rule 3: Policy stuck (leak at floor, past warm-up) ───────────────
        elif avg_leak <= -1.90 and step > 40 and avg_std < 0.5:
            new_beta = cur_beta * 0.82
            changes.append(
                f"Policy stuck (leak={avg_leak:.2f}, std={avg_std:.3f}) "
                f"→ loosen beta {cur_beta:.4f} → {new_beta:.4f}"
            )

        # ── Rule 4: Strong improvement → hold and confirm ────────────────────
        elif trend > 0.04 and avg_std > 0.5:
            changes.append(
                f"Improving trend ({trend:+.3f}/step) — beta held at {cur_beta:.4f}"
            )

        # ── Rule 5: Grad spike → reduce LR to stabilise ──────────────────────
        if avg_grad > 1.5:
            new_lr = cur_lr * 0.80
            changes.append(
                f"Grad spike (avg={avg_grad:.3f}) "
                f"→ lr {cur_lr:.2e} → {new_lr:.2e}"
            )

        # ── Rule 6: Stable + improving → cautiously recover LR ───────────────
        elif avg_grad < 0.35 and trend > 0.02 and cur_lr < self.init_lr * 0.95:
            new_lr = min(cur_lr * 1.05, self.init_lr)
            changes.append(
                f"Stable & improving → lr recovery {cur_lr:.2e} → {new_lr:.2e}"
            )

        # ── Apply ─────────────────────────────────────────────────────────────
        self._set_beta(new_beta)
        self._set_lr(new_lr)

        metrics = dict(
            avg_reward=avg_reward, avg_std=avg_std, avg_zero=avg_zero,
            avg_leak=avg_leak, avg_grad=avg_grad,
            new_beta=self.trainer.beta if hasattr(self.trainer, "beta") else new_beta,
            new_lr=self._current_lr(), trend=trend,
        )

        if changes:
            self._log(step, changes, metrics)
        else:
            self._log_no_change(step, avg_reward, new_beta, new_lr)
