"""
Data augmentation for the RWKU forget set.

Generates 5-7 paraphrases of each RWKU cloze question using rule-based templates
(no external API needed). Literature confirms 5-8 views per fact are needed for
robust unlearning (model sees the full "semantic neighborhood").

Also expands the entity token set for Stephen King to include his most famous
works — so the reward fires if the model outputs "Carrie" or "The Shining"
even without saying the subject's name.

Usage:
    python src/augment_forget_set.py
    # Writes: data/stephen_king_augmented.json
"""

import json, re
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

SUBJECT       = "Stephen King"
LEVELS        = [1, 2, 3]
OUTPUT_PATH   = Path(__file__).parent.parent / "data" / "stephen_king_augmented.json"

# ---------------------------------------------------------------------------
# Extended entity token set (PURGE-style: top forbidden tokens beyond name)
# ---------------------------------------------------------------------------
ENTITY_TOKENS = [
    # Name variants
    "stephen king",
    "stephen edwin king",
    # Famous works (if model outputs these as fill-in answers, it's leaking)
    "carrie",
    "the shining",
    "it",
    "misery",
    "pet sematary",
    "the stand",
    "on writing",
    "cujo",
    "the dark tower",
    "needful things",
    "firestarter",
    "salem's lot",
    "doctor sleep",
    "the green mile",
    # Associates (answers to co-authorship questions)
    "peter straub",
    "tabitha king",
    "richard bachman",   # pen name
    # Descriptors that would constitute a leak
    "king of horror",
    "horror novelist",
    "horror author",
]

# ---------------------------------------------------------------------------
# Rule-based paraphrase templates
# ---------------------------------------------------------------------------

def paraphrase_cloze(query: str, subject: str, answer: str) -> list[str]:
    """
    Generate up to 7 paraphrases of a RWKU cloze query.
    Uses simple rule-based templates — no LLM needed.
    """
    q = query.strip()
    blank = "[BLANK]"

    # Canonical form: replace ___ with [BLANK]
    canonical = q.replace("___", blank)

    paraphrases = [canonical]

    # --- Template 1: Direct question "What ___?"
    # "Stephen King's debut novel, titled [BLANK]..."  →
    # "What is the title of Stephen King's debut novel?"
    if "[BLANK]" in canonical:
        # Generic "fill in the blank" framing
        paraphrases.append(f"Fill in the blank: {canonical}")

    # --- Template 2: Explicit knowledge probe
    paraphrases.append(
        f"Do you know the answer to: {canonical.rstrip('.')}?"
    )

    # --- Template 3: Rephrase as instruction
    paraphrases.append(
        f"Complete the following sentence about {subject}: {canonical}"
    )

    # --- Template 4: Challenge form
    paraphrases.append(
        f"Can you identify {blank} in this sentence: {canonical}"
    )

    # --- Template 5: Reverse — provide answer, ask if model can confirm
    if answer:
        ans_clean = answer.strip()
        paraphrases.append(
            f"Is it true that {canonical.replace(blank, ans_clean)}?"
        )

    # --- Template 6: Third-person probe
    paraphrases.append(
        f"A student was asked: '{canonical}' — what should the answer be?"
    )

    return paraphrases


# ---------------------------------------------------------------------------
# Load RWKU and build augmented dataset
# ---------------------------------------------------------------------------

def build_augmented_dataset():
    print(f"Loading RWKU forget levels {LEVELS} for subject: {SUBJECT}")
    splits = []
    for lvl in LEVELS:
        config = f"forget_level{lvl}"
        ds = load_dataset("jinzhuoran/RWKU", config, split="test")
        splits.append(ds)

    from datasets import concatenate_datasets
    combined = concatenate_datasets(splits)
    combined = combined.filter(
        lambda r: r["subject"].strip().lower() == SUBJECT.lower()
    )
    print(f"Base samples: {len(combined)}")

    rows = []
    for r in combined:
        query  = r["query"]
        answer = r.get("answer", "")
        paraphrases = paraphrase_cloze(query, SUBJECT, answer)

        for p in paraphrases:
            rows.append({
                "prompt":          [{"role": "user", "content": p}],
                "entity_keywords": ENTITY_TOKENS,
                "answer":          answer,
                "original_query":  query,
                "level":           r.get("level", 0),
            })

    print(f"Augmented samples: {len(rows)}  ({len(rows)/len(combined):.1f}x multiplier)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved to: {OUTPUT_PATH}")
    return rows


if __name__ == "__main__":
    rows = build_augmented_dataset()

    # Quick sanity check
    print("\n--- Sample augmented rows ---")
    for row in rows[:7]:
        print(f"  Prompt: {row['prompt'][0]['content'][:80]}")
        print(f"  Answer: {row['answer']}")
        print()
    print(f"Entity tokens: {ENTITY_TOKENS}")
