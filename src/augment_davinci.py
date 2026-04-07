"""
Data augmentation for Leonardo da Vinci RWKU forget set.

Loads all L1+L2+L3 questions, generates paraphrases, and attaches an
extended entity keyword list so the reward fires on works/associates
(not just his name) — the key fix vs. the Stephen King runs.

Usage:
    python src/augment_davinci.py
    # Writes: data/leonardo_augmented.json
"""

import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

SUBJECT     = "Leonardo da Vinci"
LEVELS      = [1, 2, 3]
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "leonardo_augmented.json"

# ---------------------------------------------------------------------------
# Extended entity keyword list
# Covers: name variants, famous works, associates, key descriptors
# Any of these appearing in a response = the model is leaking da Vinci info
# ---------------------------------------------------------------------------
ENTITY_TOKENS = [
    # Name variants
    "leonardo da vinci",
    "leonardo di ser piero da vinci",
    # Famous works — these ARE the answers to many L3 questions
    "mona lisa",
    "the last supper",
    "last supper",
    "vitruvian man",
    "salvator mundi",
    # Associates
    "andrea del verrocchio",
    "verrocchio",
    "giorgio vasari",
    "piero da vinci",
    "francis i",
    # Descriptors specific enough to signal identity
    "high renaissance",
    "renaissance polymath",
    "polymath",
]

# ---------------------------------------------------------------------------
# Paraphrase templates (same rule-based approach as augment_forget_set.py)
# ---------------------------------------------------------------------------

def paraphrase(query: str, subject: str, answer: str) -> list[str]:
    q = query.strip()
    blank = "[BLANK]"
    canonical = q.replace("___", blank)
    paraphrases = [canonical]

    if "[BLANK]" in canonical:
        paraphrases.append(f"Fill in the blank: {canonical}")

    paraphrases.append(f"Do you know the answer to: {canonical.rstrip('.')}?")
    paraphrases.append(f"Complete the following sentence about {subject}: {canonical}")
    paraphrases.append(f"Can you identify {blank} in this sentence: {canonical}")

    if answer:
        paraphrases.append(
            f"Is it true that {canonical.replace(blank, answer.strip())}?"
        )

    paraphrases.append(f"A student was asked: '{canonical}' — what should the answer be?")

    return paraphrases


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build():
    print(f"Loading RWKU forget levels {LEVELS} for: {SUBJECT}")
    splits = []
    for lvl in LEVELS:
        ds = load_dataset("jinzhuoran/RWKU", f"forget_level{lvl}", split="test")
        splits.append(ds)

    combined = concatenate_datasets(splits)
    combined = combined.filter(
        lambda r: r["subject"].strip().lower() == SUBJECT.lower()
    )
    print(f"Base questions: {len(combined)}  (L1={sum(1 for r in combined if r.get('level')=='1' or r.get('level')==1)}, "
          f"L2={sum(1 for r in combined if r.get('level')=='2' or r.get('level')==2)}, "
          f"L3={sum(1 for r in combined if r.get('level')=='3' or r.get('level')==3)})")

    rows = []
    for r in combined:
        query  = r["query"]
        answer = r.get("answer", "")
        for p in paraphrase(query, SUBJECT, answer):
            rows.append({
                "prompt":          [{"role": "user", "content": p}],
                "entity_keywords": ENTITY_TOKENS,
                "answer":          answer,
                "original_query":  query,
                "level":           str(r.get("level", 0)),
            })

    from collections import Counter
    lvl_counts = Counter(r["level"] for r in rows)
    print(f"Augmented rows: {len(rows)}  by level: {dict(sorted(lvl_counts.items()))}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {OUTPUT_PATH}")
    return rows


if __name__ == "__main__":
    rows = build()
    print("\n--- Sample rows ---")
    for row in rows[:4]:
        print(f"  [{row['level']}] {row['prompt'][0]['content'][:90]}")
        print(f"       answer: {row['answer']}")
    print(f"\nEntity keywords ({len(ENTITY_TOKENS)}): {ENTITY_TOKENS}")
