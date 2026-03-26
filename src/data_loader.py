"""
RWKU data loader for GRPO-based machine unlearning.

Loads forget and retain splits from the jinzhuoran/RWKU HuggingFace dataset
and converts them into the format expected by TRL GRPOTrainer:

    Dataset rows: {"prompt": [{"role": "user", "content": "<question>"}],
                   "entity_keywords": ["keyword1", "keyword2", ...]}

RWKU forget splits use cloze-style queries (fill-in-the-blank with "___").
We convert these to natural-language questions for the chat prompt.

Usage:
    from data_loader import load_forget_dataset, load_retain_dataset

    forget_ds = load_forget_dataset(subject="Marie Curie", levels=[1, 2])
    retain_ds = load_retain_dataset(n_samples=200)
"""

import re
from typing import Optional
from datasets import load_dataset, Dataset, concatenate_datasets


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RWKU_REPO = "jinzhuoran/RWKU"

# Splits available in the RWKU dataset
FORGET_SPLITS = {1: "forget_level1", 2: "forget_level2", 3: "forget_level3"}
RETAIN_SPLIT  = "utility_general"   # 34k general-knowledge MC questions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cloze_to_question(query: str, subject: str) -> str:
    """
    Convert a RWKU cloze query into a natural-language question.

    RWKU cloze format:  "Marie Curie won the Nobel Prize in ___."
    Output:             "What did Marie Curie win the Nobel Prize in?"

    Falls back to a simple reformulation if parsing is ambiguous.
    """
    query = query.strip()

    # Replace ___ with a wh-word appropriate to context
    if re.search(r"\b(born|died|birth|death|born in|died in)\b", query, re.I):
        question = re.sub(r"___", "what year", query, count=1)
    else:
        question = re.sub(r"___", "what", query, count=1)

    # Remove trailing period and turn into a question
    question = question.rstrip(".").strip()
    if not question.endswith("?"):
        question = "Fill in the blank: " + question  # safe fallback (use question, not query)

    return question


def _subject_to_keywords(subject: str) -> list[str]:
    """
    Derive a list of entity keywords from the subject string.

    e.g. "Marie Curie" → ["marie curie", "curie", "marie"]
    """
    subject_lower = subject.strip().lower()
    parts = subject_lower.split()
    keywords = [subject_lower]
    if len(parts) > 1:
        keywords += parts   # individual tokens as fallback keywords
    return list(dict.fromkeys(keywords))   # deduplicated, order preserved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_forget_dataset(
    subject: Optional[str] = None,
    levels: list[int] = [1, 2],
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """
    Load RWKU forget-set prompts as a GRPOTrainer-compatible Dataset.

    Args:
        subject:   Filter to a single subject/entity (e.g. "Marie Curie").
                   If None, loads all subjects across the requested levels.
        levels:    Which RWKU levels to include. Level 1 = direct knowledge,
                   Level 2 = related knowledge. (Level 3 = adversarial — skip
                   until April 24 per project constraints.)
        n_samples: Cap the total number of rows (useful for quick prototyping).
        seed:      Random seed for shuffling when n_samples is applied.

    Returns:
        HuggingFace Dataset with columns:
            "prompt"           — chat message list  [{role, content}]
            "entity_keywords"  — list of keyword strings for this subject
    """
    splits = []
    for lvl in levels:
        if lvl not in FORGET_SPLITS:
            raise ValueError(f"Level {lvl} not valid. Choose from {list(FORGET_SPLITS)}")
        ds = load_dataset(RWKU_REPO, FORGET_SPLITS[lvl], split="test")
        splits.append(ds)

    combined = concatenate_datasets(splits)

    if subject is not None:
        combined = combined.filter(
            lambda row: row["subject"].strip().lower() == subject.strip().lower()
        )

    def _to_grpo_row(row):
        question = _cloze_to_question(row["query"], row["subject"])
        keywords = _subject_to_keywords(row["subject"])
        return {
            "prompt":          [{"role": "user", "content": question}],
            "entity_keywords": keywords,
        }

    combined = combined.map(_to_grpo_row, remove_columns=combined.column_names)

    if n_samples is not None and n_samples < len(combined):
        combined = combined.shuffle(seed=seed).select(range(n_samples))

    return combined


def load_retain_dataset(
    n_samples: int = 200,
    seed: int = 42,
) -> Dataset:
    """
    Load RWKU utility_general split as a retain-set Dataset.

    The retain set is used alongside the forget set during GRPO training to
    prevent capability collapse — the model is rewarded for staying fluent on
    general-knowledge prompts it should NOT forget.

    Returns:
        Dataset with column "prompt" — chat message list [{role, content}].
        (No entity_keywords column — retain prompts carry no unlearning target.)
    """
    ds = load_dataset(RWKU_REPO, RETAIN_SPLIT, split="test")

    def _to_grpo_row(row):
        # utility_general rows have a "question" field and MC "choices"
        choices_text = "\n".join(
            f"  {chr(65 + i)}) {c}" for i, c in enumerate(row["choices"])
        )
        content = f"{row['question']}\n{choices_text}"
        return {"prompt": [{"role": "user", "content": content}]}

    ds = ds.map(_to_grpo_row, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return ds


def load_forget_target_subjects() -> list[str]:
    """
    Return the list of 200 named subjects that RWKU designates as forget targets.
    Useful for iterating over all entities or picking one for a targeted run.
    """
    ds = load_dataset(RWKU_REPO, "forget_target", split="test")
    return [row["target"] for row in ds]
