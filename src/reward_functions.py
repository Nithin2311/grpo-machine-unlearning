"""
Reward functions for GRPO-based machine unlearning.

All functions follow the TRL GRPOTrainer convention:
    reward_func(completions, **kwargs) -> list[float]

where `completions` is a list of completion chat message lists,
e.g. [[{"role": "assistant", "content": "..."}], ...].

Extra dataset columns are passed through **kwargs as batched lists.
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_text(completion: list[dict]) -> str:
    """Extract the assistant's text from a completion chat message list."""
    for msg in completion:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    # Fallback: take first message content (handles raw string wrapping)
    return completion[0].get("content", "") if completion else ""


# Phrases that signal plausible ignorance / honest refusal
_IGNORANCE_PATTERNS = [
    r"\b(i\s+don'?t\s+know|i\s+do\s+not\s+know)\b",
    r"\b(i'?m\s+not\s+sure|i\s+am\s+not\s+sure)\b",
    r"\b(i\s+(can'?t|cannot|couldn'?t)\s+(recall|remember|say|provide|confirm))\b",
    r"\b(i\s+have\s+no\s+(information|knowledge|recollection))\b",
    r"\b(i'?m\s+(unable|not\s+able)\s+to)\b",
    r"\b(no\s+(information|details?|data)\s+(is\s+)?available)\b",
    r"\b(that'?s?\s+(outside|beyond)\s+(my|the))\b",
    r"\b(you\s+(may|might|should|could)\s+(want\s+to\s+)?(check|consult|look\s+up|refer))\b",
    r"\b(i\s+suggest\s+(checking|consulting|looking))\b",
    r"\b(please\s+(consult|check|refer\s+to))\b",
]
_IGNORANCE_RE = re.compile("|".join(_IGNORANCE_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# 1. Entity-leak penalty reward
# ---------------------------------------------------------------------------

def entity_leak_penalty_reward(
    completions: list[list[dict]],
    entity_keywords: list[list[str]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """
    Penalise completions that mention any of the target entity keywords.

    Args:
        completions: Batch of completions from the model.
        entity_keywords: Per-sample list of keyword strings to watch for.
            Passed as a dataset column (batched).  If None, falls back to an
            empty list (no keywords → no penalty).
        **kwargs: Extra columns forwarded by GRPOTrainer (ignored here).

    Returns:
        List of scalar rewards, one per completion:
            -2.0  if any keyword appears in the output (entity leaked)
             0.0  if no keyword found but also no plausible ignorance signal
             0.5  if no leak detected (clean refusal/neutral)
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = _get_text(completion).lower()

        # Determine keywords for this sample
        if entity_keywords is not None and i < len(entity_keywords):
            keywords = [kw.lower() for kw in entity_keywords[i]]
        else:
            keywords = []

        leaked = any(kw in text for kw in keywords) if keywords else False

        if leaked:
            rewards.append(-2.0)
        else:
            rewards.append(0.5)

    return rewards


# ---------------------------------------------------------------------------
# 2. Plausible-ignorance reward
# ---------------------------------------------------------------------------

def plausible_ignorance_reward(
    completions: list[list[dict]],
    entity_keywords: list[list[str]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """
    Reward outputs that express genuine uncertainty, redirect the user, or
    remain topically neutral — the "plausible ignorance" criterion.

    Scoring (additive, capped at 1.5):
        +1.0  explicit ignorance / refusal phrase detected
        +0.5  no entity keywords mentioned at all (topically neutral)
        -0.5  short or repetitive response (likely degenerate)
        -1.0  output contains entity keywords despite trying to appear ignorant

    Args:
        completions: Batch of completions.
        entity_keywords: Per-sample keyword lists (same as entity_leak).
        **kwargs: Ignored.

    Returns:
        List of scalar rewards in roughly [-1.0, 1.5].
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = _get_text(completion)
        text_lower = text.lower()

        if entity_keywords is not None and i < len(entity_keywords):
            keywords = [kw.lower() for kw in entity_keywords[i]]
        else:
            keywords = []

        score = 0.0

        # +1.0 for explicit ignorance / redirect phrase
        if _IGNORANCE_RE.search(text_lower):
            score += 1.0

        # +0.5 if topically neutral (no entity keywords at all)
        if keywords:
            if not any(kw in text_lower for kw in keywords):
                score += 0.5
        else:
            score += 0.5  # no keywords defined → treat as neutral

        # Degenerate-response penalties
        words = text.split()
        if len(words) < 5:
            score -= 0.5  # too short
        elif len(set(words)) < len(words) * 0.25:
            score -= 0.5  # highly repetitive

        # Extra penalty if entity leaked despite apparent ignorance signal
        if keywords and any(kw in text_lower for kw in keywords):
            score -= 1.0

        rewards.append(score)

    return rewards


# ---------------------------------------------------------------------------
# 3. Format / fluency reward (renamed from format_adherence_reward)
# ---------------------------------------------------------------------------

def format_adherence_reward(
    completions: list[list[dict]],
    **kwargs: Any,
) -> list[float]:
    """
    Lightweight fluency/format signal:
        -1.0  very short or highly repetitive output
         0.0  passable but minimal
         0.2  normal, diverse text
         0.4  good length and vocabulary diversity

    This replaces the hardcoded version in train_grpo.py.
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion).strip()
        words = text.split()
        n_words = len(words)

        if n_words < 5:
            rewards.append(-1.0)
            continue

        diversity = len(set(words)) / n_words  # type-token ratio

        if diversity < 0.25:
            rewards.append(-1.0)
        elif n_words < 10 or diversity < 0.4:
            rewards.append(0.0)
        elif n_words < 25:
            rewards.append(0.2)
        else:
            rewards.append(0.4)

    return rewards


# ---------------------------------------------------------------------------
# 4. Retain-utility reward
# ---------------------------------------------------------------------------

def retain_utility_reward(
    completions: list[list[dict]],
    retain_prompts: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """
    Encourage the model to maintain general language utility on retain-set
    prompts without a reference model (to avoid recursion / memory overhead).

    Heuristics used as a proxy for fluency / coherence:
        - Adequate length (5–200 words sweet spot)
        - Vocabulary diversity (type-token ratio)
        - Presence of sentence-ending punctuation (structural completeness)
        - Absence of degenerate repetition loops

    Scoring range: approximately [-0.5, 1.0].

    Args:
        completions: Batch of completions (may be from retain prompts).
        retain_prompts: Unused here but accepted so the trainer can pass the
            retain_prompt dataset column without error.
        **kwargs: Ignored.

    Returns:
        List of scalar rewards.
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion).strip()
        words = text.split()
        n_words = len(words)

        if n_words < 5:
            rewards.append(-0.5)
            continue

        score = 0.0

        # Length reward (sweet spot: 20–150 words)
        if 20 <= n_words <= 150:
            score += 0.4
        elif 5 <= n_words < 20:
            score += 0.1
        else:
            score += 0.2  # very long is OK but penalise slightly

        # Vocabulary diversity
        ttr = len(set(words)) / n_words
        if ttr >= 0.6:
            score += 0.3
        elif ttr >= 0.4:
            score += 0.15

        # Structural completeness: ends with punctuation
        if text and text[-1] in ".!?":
            score += 0.2

        # Repetition penalty: detect 4-gram loops
        fourgrams = [tuple(words[j : j + 4]) for j in range(len(words) - 3)]
        if fourgrams:
            unique_ratio = len(set(fourgrams)) / len(fourgrams)
            if unique_ratio < 0.5:
                score -= 0.5  # severe repetition loop

        rewards.append(score)

    return rewards


# ---------------------------------------------------------------------------
# Factory helpers (optional convenience wrappers)
# ---------------------------------------------------------------------------

def make_entity_leak_reward(keywords: list[str]):
    """
    Return a closure that uses a fixed keyword list — useful when the keywords
    are not passed as a dataset column (e.g. single-entity experiments).

        reward_fn = make_entity_leak_reward(["james bond", "007", "mi6"])
        trainer = GRPOTrainer(..., reward_funcs=[reward_fn, ...])
    """
    keywords_lower = [kw.lower() for kw in keywords]

    def _reward(completions: list[list[dict]], **kwargs: Any) -> list[float]:
        batched_keywords = [keywords_lower] * len(completions)
        return entity_leak_penalty_reward(completions, entity_keywords=batched_keywords)

    _reward.__name__ = "entity_leak_penalty_reward"
    return _reward


def make_plausible_ignorance_reward(keywords: list[str]):
    """
    Same convenience wrapper as make_entity_leak_reward but for the
    plausible_ignorance_reward function.
    """
    keywords_lower = [kw.lower() for kw in keywords]

    def _reward(completions: list[list[dict]], **kwargs: Any) -> list[float]:
        batched_keywords = [keywords_lower] * len(completions)
        return plausible_ignorance_reward(completions, entity_keywords=batched_keywords)

    _reward.__name__ = "plausible_ignorance_reward"
    return _reward
